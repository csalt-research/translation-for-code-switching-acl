# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# set -e

#
# Data preprocessing configuration
#

N_MONO=10000000  # number of monolingual sentences for each language
CODES=60000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data

MONO_PATH5=$DATA_PATH/hindiblogcs
MONO_PATH6=$DATA_PATH/treebankmono

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fastBPE/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
CONCAT_BPE=$DATA_PATH/all

SRC_VOCAB=$DATA_PATH/vocab.hi
TGT_VOCAB=$DATA_PATH/vocab.en
FULL_VOCAB=$DATA_PATH/vocab.all

SRC_RAW13=$MONO_PATH5/hi.train
SRC_RAW14=$MONO_PATH5/hi.test
SRC_RAW15=$MONO_PATH5/hi.valid

TGT_RAW13=$MONO_PATH5/en.train
TGT_RAW14=$MONO_PATH5/en.test
TGT_RAW15=$MONO_PATH5/en.valid

SRC_RAW16=$MONO_PATH6/hi.train
SRC_RAW17=$MONO_PATH6/hi.test
SRC_RAW18=$MONO_PATH6/hi.valid

TGT_RAW16=$MONO_PATH6/en.train
TGT_RAW17=$MONO_PATH6/en.test
TGT_RAW18=$MONO_PATH6/en.valid

# SRC_TEST=$PARA_PATH/opus_hi.test
# TGT_TEST=$PARA_PATH/opus_en.test


#
# Download and install tools
#

# # Download Moses
# cd $TOOLS_PATH
# if [ ! -d "$MOSES" ]; then
#   echo "Cloning Moses from GitHub repository..."
#   git clone https://github.com/moses-smt/mosesdecoder.git
# fi
# echo "Moses found in: $MOSES"

# # Download fastBPE
# cd $TOOLS_PATH
# if [ ! -d "$FASTBPE_DIR" ]; then
#   echo "Cloning fastBPE from GitHub repository..."
#   git clone https://github.com/glample/fastBPE
# fi
# echo "fastBPE found in: $FASTBPE_DIR"

# # Compile fastBPE
# cd $TOOLS_PATH
# if [ ! -f "$FASTBPE" ]; then
#   echo "Compiling fastBPE..."
#   cd $FASTBPE_DIR/fastBPE
#   g++ -std=c++11 -pthread -O3 main.cc -o fast
# fi
# echo "fastBPE compiled in: $FASTBPE"

# # Download fastText
# cd $TOOLS_PATH
# if [ ! -d "$FASTTEXT_DIR" ]; then
#   echo "Cloning fastText from GitHub repository..."
#   git clone https://github.com/facebookresearch/fastText.git
# fi
# echo "fastText found in: $FASTTEXT_DIR"

# # Compile fastText
# cd $TOOLS_PATH
# if [ ! -f "$FASTTEXT" ]; then
#   echo "Compiling fastText..."
#   cd $FASTTEXT_DIR
#   make
# fi
# echo "fastText compiled in: $FASTTEXT"


# echo "Tokenize monolingual data..."
# cat $SRC_RAW1 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_TOK1
# cat $SRC_RAW2 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_TOK2
# cat $SRC_RAW3 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_TOK3
# cat $SRC_RAW4 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_TOK4

# cat $TGT_RAW1 | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TOK1
# cat $TGT_RAW2 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TOK2
# cat $TGT_RAW3 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TOK3
# cat $TGT_RAW4 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TOK4

# echo "HI monolingual data tokenized in: $SRC_TOK"
# echo "EN monolingual data tokenized in: $TGT_TOK"

# # learn BPE codes
# if [ ! -f "$BPE_CODES" ]; then
#   echo "Learning BPE codes..."
#   $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
# fi
# echo "BPE learned in $BPE_CODES"

# # apply BPE codes
# if ! [[ -f "$SRC_TOK.$CODES" && -f "$TGT_TOK.$CODES" ]]; then
#   echo "Applying BPE codes..."
#   $FASTBPE applybpe $SRC_TOK.$CODES $SRC_TOK $BPE_CODES
#   $FASTBPE applybpe $TGT_TOK.$CODES $TGT_TOK $BPE_CODES
# fi
# echo "BPE codes applied to HI in: $SRC_TOK.$CODES"
# echo "BPE codes applied to EN in: $TGT_TOK.$CODES"

# extract vocabulary
# if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" && -f "$FULL_VOCAB" ]]; then
#   echo "Extracting vocabulary..."
#   $FASTBPE getvocab $SRC_TOK.$CODES > $SRC_VOCAB
#   $FASTBPE getvocab $TGT_TOK.$CODES > $TGT_VOCAB
#   $FASTBPE getvocab $SRC_TOK.$CODES $TGT_TOK.$CODES > $FULL_VOCAB
# fi
# echo "HI vocab in: $SRC_VOCAB"
# echo "EN vocab in: $TGT_VOCAB"
# echo "Full vocab in: $FULL_VOCAB"

# echo "Extracting vocabulary..."
# cat $SRC_RAW2 $SRC_RAW4 $SRC_RAW7 $SRC_RAW10 | $FASTBPE getvocab - >$SRC_VOCAB
# cat $TGT_RAW2 $TGT_RAW4 $TGT_RAW7 $TGT_RAW10 | $FASTBPE getvocab - >$TGT_VOCAB
# cat $SRC_RAW2 $TGT_RAW2 $SRC_RAW4 $TGT_RAW4 $SRC_RAW7 $TGT_RAW7 $SRC_RAW10 $TGT_RAW10 | $FASTBPE getvocab - >$FULL_VOCAB

# $FASTBPE getvocab $SRC_TOK > $SRC_VOCAB
# $FASTBPE getvocab $TGT_TOK > $TGT_VOCAB
# $FASTBPE getvocab $SRC_TOK $TGT_TOK > $FULL_VOCAB
# fi

echo "HI vocab in: $SRC_VOCAB"
echo "EN vocab in: $TGT_VOCAB"
echo "Full vocab in: $FULL_VOCAB"

# binarize data
# if ! [[ -f "$SRC_TOK.$CODES.pth" && -f "$TGT_TOK.$CODES.pth" ]]; then
#   echo "Binarizing data..."
#   $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TOK.$CODES
#   $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TOK.$CODES
# fi
# echo "EN binarized data in: $SRC_TOK.$CODES.pth"
# echo "FR binarized data in: $TGT_TOK.$CODES.pth"

# if ! [[ -f "$SRC_TOK.pth" && -f "$TGT_TOK.pth" ]]; then
echo "Binarizing data..."
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW1
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW2
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW3

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW4
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW5
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW6

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW1
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW2
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW3

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW4
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW5
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW6

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW7
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW8
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW9

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW7
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW8
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW9

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW10
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW11
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW12

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW10
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW11
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW12

python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW16
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW17
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW18

python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW16
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW17
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW18

python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW13
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW14
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW15

python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW13
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW14
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW15

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW19
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW20
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW21

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW19
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW20
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW21

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW19
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW20
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW21

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW13
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW14
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW15


# echo "Tokenizing valid and test data..."
# cat $SRC_VALID1 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_VALID_TOK1
# cat $SRC_VALID2 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_VALID_TOK2
# cat $SRC_VALID3 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_VALID_TOK3
# cat $SRC_VALID4 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_VALID_TOK4

# cat $TGT_VALID1 | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_VALID_TOK1
# cat $TGT_VALID2 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_VALID_TOK2
# cat $TGT_VALID3 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_VALID_TOK3
# cat $TGT_VALID4 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_VALID_TOK4

# cat $SRC_TEST1 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_TEST_TOK1
# cat $SRC_TEST2 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_TEST_TOK2
# cat $SRC_TEST3 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_TEST_TOK3
# cat $SRC_TEST4 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $SRC_TEST_TOK4

# cat $TGT_TEST1 | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $TGT_TEST_TOK1
# cat $TGT_TEST2 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TEST_TOK2
# cat $TGT_TEST3 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TEST_TOK3
# cat $TGT_TEST4 | $NORM_PUNC -l hi | $TOKENIZER -l hi -no-escape -threads $N_THREADS > $TGT_TEST_TOK4

# echo "Applying BPE to valid and test files..."
# $FASTBPE applybpe $SRC_VALID.$CODES $SRC_VALID $BPE_CODES $SRC_VOCAB
# $FASTBPE applybpe $TGT_VALID.$CODES $TGT_VALID $BPE_CODES $TGT_VOCAB
# $FASTBPE applybpe $SRC_TEST.$CODES $SRC_TEST $BPE_CODES $SRC_VOCAB
# $FASTBPE applybpe $TGT_TEST.$CODES $TGT_TEST $BPE_CODES $TGT_VOCAB

# echo "Binarizing data..."
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID1
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID2
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID3
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID4

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID1
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID2
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID3
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID4

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST1
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST2
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST3
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST4

# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST1
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST2
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST3
# python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST4

# echo "Concatenating source and target monolingual data..."
# cat $SRC_RAW1 $SRC_RAW2 $SRC_RAW3 $SRC_RAW4 $SRC_RAW5 $SRC_RAW6 $SRC_RAW7 $SRC_RAW8 $SRC_RAW9 $SRC_RAW10 $SRC_RAW11 $SRC_RAW12 $TGT_RAW1 $TGT_RAW2 $TGT_RAW3 $TGT_RAW4 $TGT_RAW5 $TGT_RAW6 $TGT_RAW7 $TGT_RAW8 $TGT_RAW9 $TGT_RAW10 $TGT_RAW11 $TGT_RAW12 | shuf > $CONCAT_BPE
# # fi
# echo "Concatenated data in: $CONCAT_BPE"

# # if ! [[ -f "$CONCAT_BPE.vec" ]]; then
# echo "Training fastText on $CONCAT_BPE..."
# $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 256 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE".256"
# $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE".512"
# # fi
# # echo "Cross-lingual embeddings in: $CONCAT_BPE.vec"



