# process command line arguments (support for flags)
while getopts d:t:f: flag
do
    case "${flag}" in
        d) dataset=../${OPTARG};;
        t) threads=${OPTARG};;
        f) topic_file=${OPTARG};; 
    esac
done

# set defaults if arguments are not passed
dataset="${dataset:-../dblp}"
threads="${threads:-20}"
topic_file="${topic_file:-field}"

# print settable parameters
echo "Launching Partial Concept Learning"
echo "    Dataset:           ${dataset}"
echo "    Threads:           ${threads}"
echo "    Topic File Suffix: ${topic_file}"

# text file name; one document per line
text_file=sentences_.txt

# topic file suffix
output_file=${topic_file}

emb_file=emb_part_${output_file}

make train_emb_part_tax

# default:
#     reg_lambda = 10
#     global_lambda = 1.5
#     window = 5
#     pretrain = 5
./train_emb_part_tax -train ./${dataset}/${text_file} -output ./${dataset}/${emb_file}_w.txt -kappa ./${dataset}/${emb_file}_cap.txt -topic ./${dataset}/topics_${topic_file}.txt -topic_output ./${dataset}/${emb_file}_t.txt -doc_output ./${dataset}/${emb_file}_s.txt -reg_lambda 1 -cbow 0 -size 100 -global_lambda 1.5 -window 5 -negative 5 -sample 1e-3 -min-count 50 -threads ${threads} -binary 0 -iter 10 -pretrain 5 -rank_product 0 -gen_vocab 0 -load_emb 0

		