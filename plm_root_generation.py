"""
    File with all of the functions required to rank and generate likely root nodes for the
    first level of the given taxonomy
"""
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
from torch.nn import CrossEntropyLoss
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

def create_prompt(prompt_txt, tokenizer):
    """Create a function which given a child and parent entity, generates a 
    sentence string for the model to generate a loss for

    Args:
        prompt_txt (str): String to use as the basis of the prompt, where "__child__"
            is replaced with the child passed to the function and where "__parent__"
            is replace with the parent passed to the function
        tokenizer (GPT2TokenizerFast): GPT-2 Tokenizer

    Raises:
        ValueError: On invalid prompts.

    Returns:
        fxn: Function with parameters (child, parent) with the behavior described above.
    """

    # Data validation
    if '__child__' not in prompt_txt:
        raise ValueError(f'The prompt: "{prompt_txt}" does not contain "__child__"')
    elif '__parent__' not in prompt_txt:
        raise ValueError(f'The prompt: "{prompt_txt}" does not contain "__parent__"')
    elif '.' not in prompt_txt:
        raise ValueError(f'The prompt: "{prompt_txt}" does not contain "."')

    # Function to output sentences following the prompt template
    def fxn(child, parent): 
        child, parent = child.replace('_', ' '), parent.replace('_', ' ')
        return f"{tokenizer.bos_token} {prompt_txt.replace('__child__', child).replace('__parent__', parent)}"

    # Return created function
    return fxn

def load_prompts(prompt_filepath, tokenizer):
    """Create an array of prompt function from a prompt template file

    Args:
        prompt_filepath (str): File path to prompt templates.
        tokenizer (GPT2TokenizerFast): GPT-2 Tokenizer.

    Returns:
        list: List of sentence-generating functions.
    """
    # List for the generated functions
    prompts = []

    # Iterate through prompt strings, creating the prompts
    with open(prompt_filepath, 'r') as f:
        for line in f:
            prompt_txt = line.strip()
            prompts.append(create_prompt(prompt_txt, tokenizer))

    # Return sentence generation functions
    return prompts

def compute_loss(
    model,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    """Modified GPT-2 loss computation function.

    Returns:
        list: Losses for the sentences passed in.
    """
    # Run inputs through the transformer
    return_dict = return_dict if return_dict is not None else model.config.use_return_dict
    transformer_outputs = model.transformer(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    hidden_states = transformer_outputs[0]
    
    # Set device for model parallelism
    if model.model_parallel:
        torch.cuda.set_device(model.transformer.first_device)
        hidden_states = hidden_states.to(model.lm_head.weight.device)
    
    # Latent model output
    lm_logits = model.lm_head(hidden_states)
    
    # Losses for the pad-separated input text spans
    loss_list = []

    # Shift so that tokens < n predict n
    shift_logits_list = lm_logits[..., :-1, :].contiguous()
    shift_labels_list = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='sum')
    for shift_logits, shift_labels in zip(shift_logits_list, shift_labels_list):
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.detach().cpu().numpy()
        loss_list.append(loss)

    return loss_list

def score_batch(sents, model, tokenizer, device):
    """Compute the losses for a batch of sentences.

    Args:
        sents (list): List of strings. These are the sentences to compute losses for, in parallel.
        model (GPT2LMHeadModel): GPT-2 instance.
        tokenizer (GPT2TokenizerFast): GPT-2 Tokenizer.
        device (str): What device to perform torch computations on.

    Returns:
        list: List of losses for the sentence generated from GPT-2.
    """
    # Tokenize the input sentences
    tokenizer_output = tokenizer(sents)
    input_ids_list = tokenizer_output['input_ids']
    label_ids_list = deepcopy(input_ids_list)
    attention_masks_list = tokenizer_output['attention_mask']
    maxlen = max([len(input_ids) for input_ids in input_ids_list])
    
    # Modify inputs to be properly padded
    for label_ids, input_ids, attention_masks in zip(label_ids_list, input_ids_list, attention_masks_list):
        padlen = maxlen - len(input_ids)
        input_ids += [0] * padlen
        label_ids += [-100] * padlen
        attention_masks += [0] * padlen
    
    # Instantiate tensors, calculate loss on device
    try:
        # Insantiate tensors
        label_ids_tensor = torch.tensor(label_ids_list, device=device)
        input_ids_tensor = torch.tensor(input_ids_list, device=device)
        attention_masks_tensor = torch.tensor(attention_masks_list, device=device)
        
        # Calculate loss
        loss_list = compute_loss(model, input_ids_tensor, attention_mask=attention_masks_tensor, labels=label_ids_tensor)
    except (RuntimeError, KeyboardInterrupt)  as err:
        print('Attempting to free GPU memory!')
        if label_ids_tensor is not None: 
            print('\tDeleting label_ids_tensor')
            del label_ids_tensor
        if input_ids_tensor is not None: 
            print('\tDeleting input_ids_tensor')
            del input_ids_tensor
        if attention_masks_tensor is not None: 
            print('\tDeleting attention_masks_tensor')
            del attention_masks_tensor
        if device == 'cuda': 
            print('\tEmptying CUDA cache')
            torch.cuda.empty_cache()
        raise err
    else:
        # Delete tensors from the target device memory
        del label_ids_tensor
        del input_ids_tensor
        del attention_masks_tensor
    
    # Return the losses for the passed sentences
    return loss_list


def root_node_inference(first_level_topics, root_node_candidates, prompt_filepath, k=20, agg_strat='mean', max_spaces=30):
    """Find the top k root nodes for the first level topics

    Args:
        first_level_topics (list): The first level topics of the given seed taxonomy.
        root_node_candidates (list): Candidate root nodes to rank.
        prompt_filepath (str): Filepath to the text file with defined prompts.
        k (int, optional): The number of root nodes to return. Defaults to 10.
        agg_strat (str, optional): Aggregation strategy between the given prompts. Defaults to 'mean'.
            Should be one of 'mean', 'min', 'max'
        max_spaces (int, optional): Maximum number of space chars in each batch. Defaults to 30.

    Returns:
        top_nodes (list): List of length k for the top root nodes.
    """
    # Define device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-xl')

    # Load pre-trained model
    with torch.no_grad():
        model = GPT2LMHeadModel.from_pretrained('gpt2-xl').to(device)
        model.eval()

    # Load the prompts used for scoring
    prompts = load_prompts(prompt_filepath, tokenizer)

    # Instantiate list of sentences to process, as well as a 3D array for scores
    sents_to_process = []
    results = np.zeros((len(first_level_topics), len(root_node_candidates), len(prompts)))
    for x, y, z in np.ndindex(results.shape):
        sent = prompts[z](first_level_topics[x], root_node_candidates[y])
        sents_to_process.append((
            z,               # prompt_index
            x,               # first_level_topic_index
            y,               # root_candidate_index
            sent.count(' '), # space_count
            sent             # sentence
        ))

    # Sort the list
    sents_to_process.sort(key = lambda t: t[3])

    # Process sentences, following the max_spaces bound
    spaces = 0
    b_s, b_e = 0, 0
    num_sents_to_process = len(sents_to_process)
    for i, sent_tpl in enumerate(tqdm(sents_to_process)):
        # Check to see if the current batch still has space
        if (spaces := spaces + sent_tpl[3]) < max_spaces:
            if (b_e := i+1) != num_sents_to_process: continue
            
        # Get losses
        losses = score_batch([sents_to_process[s_idx][4] for s_idx in range(b_s, b_e)], model, tokenizer, device)

        # Populate the result matrix
        for s_idx in range(b_s, b_e):
            res_tpl = sents_to_process[s_idx]
            results[res_tpl[1], res_tpl[2], res_tpl[0]] = losses[s_idx-b_s]

        # Reset batch start, batch end indices
        b_s, b_e = i, i
        spaces = 0

    # Aggregation function
    agg_fxn = {
        'mean': np.mean,
        'min' : np.min,
        'max' : np.max
    }[agg_strat]

    # Aggregate across prompts
    results = agg_fxn(results, axis=2)

    # Aggregate across first layer topics
    results = agg_fxn(results, axis=0)

    # Return the top k results
    return [root_node_candidates[i] for i in np.argsort(results)[:k]]
