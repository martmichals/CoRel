#!/bin/bash

dataset=softwarestack
num_thread=127
multi=0.75
single=0.65

# ----Run This Before Setting Autophrase Threshold-----
# python3 extractCorpus.py ${dataset} ${num_thread}
# cd AutoPhrase0/
# bash auto_phrase_cp.sh ${dataset}

# ----Run This After Setting Autophrase Threshold-----
# cd AutoPhrase0/
# echo "Phrasal segmentation of the corpus"
# bash phrasal_segmentation_cp.sh ${dataset} corpus.txt ${multi} ${single}
# cp models/${dataset}/phrase_dataset_${multi}_${single}.txt ../${dataset}/phrase_text.txt
# echo "Phrasal segmentation of the sentences document"
# bash phrasal_segmentation_cp.sh ${dataset} sentences.txt ${multi} ${single}
# cp models/${dataset}/segmentation.txt ../${dataset}/
# cd ../
python3 extractSegmentation.py ${dataset}
python3 extractBertEmbedding.py ${dataset} 20
