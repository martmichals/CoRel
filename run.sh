# Arguments
#   -t/--threads:   how many cpu threads may be launched
#   -g/--gpu:       which gpu is free for use
#   -s/--seed_file: which seed file to use, filename must match topics_[a-zA-Z0-9]*.txt regex
#   -d/--dataset:   which dataset to use

# -- Parsing of input arguments -- 
# Heavily based on the following StackOverflow post: https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash 

# Parse all known arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            threads="$2"
            shift # past argument
            shift # past value
            ;;
        -g|--gpu)
            gpu="$2"
            shift # past argument
            shift # past value
            ;;
        -s|--seed_file)
            seed_file="$2"
            shift # past argument
            shift # past value
            ;;
        -d|--dataset)
            dataset="$2"
            shift # past argument
            shift # past value
            ;;
        -*|--*)
            echo "Unknown option $1"
            exit 1
            ;;
        *)
    esac
done

# Set defaults
threads="${threads:-20}"
gpu="${gpu:-0}"
seed_file="${seed_file:-field}"
dataset="${dataset:-dblp}"

# Print out parsed arguments
echo "Parsed run arguments!"
echo "    Threads to use  : $threads"
echo "    GPU Selected    : $gpu"
echo "    Seed to use     : topics_$seed_file.txt"
echo "    Dataset Selected: $dataset"

# -- Concept learning on the passed seed --

cd c
bash run_emb_part_tax.sh -d ${dataset} -t ${threads} -f ${seed_file}
cd ..

# -- Relation transferring -- 

# python3 main.py --dataset ${dataset} --topic_file topics_${seed_file}.txt --gpu ${gpu}

# -- Concept learning bash generation -- 

# -- Concept learning on the output of the relation transferring step -- 

# -- Post-processing of the final results into a single taxonomy description file -- 

# --  Generation of a visualization of the output taxonomy --
