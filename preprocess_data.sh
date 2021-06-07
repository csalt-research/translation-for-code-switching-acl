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

# Download and install tools


# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR/fastBPE
  g++ -std=c++11 -pthread -O3 main.cc -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"


DATA_PATH=$PWD/data

MONO_PATH1=$DATA_PATH/PRETRAIN_COMBINED
MONO_PATH2=$DATA_PATH/OPUS-EMT
MONO_PATH3=$DATA_PATH/OPUS-LEX
MONO_PATH4=$DATA_PATH/combinedcs

# Hi and En/CS training files

SRC_RAW1=$MONO_PATH1/hi.train
SRC_TOK1=$MONO_PATH1/hi.train.tok

SRC_RAW2=$MONO_PATH2/hi.train
SRC_TOK2=$MONO_PATH2/hi.train.tok

SRC_RAW3=$MONO_PATH3/hi.train
SRC_TOK3=$MONO_PATH3/hi.train.tok

SRC_RAW4=$MONO_PATH4/combinedcs.hi.train
SRC_TOK4=$MONO_PATH4/combinedcs.hi.train.tok

TGT_RAW1=$MONO_PATH1/en.train
TGT_TOK1=$MONO_PATH1/en.train.tok

TGT_RAW2=$MONO_PATH2/en.train
TGT_TOK2=$MONO_PATH2/en.train.tok

TGT_RAW3=$MONO_PATH3/en.train
TGT_TOK3=$MONO_PATH3/en.train.tok

TGT_RAW4=$MONO_PATH4/combinedcs.hi.train
TGT_TOK4=$MONO_PATH4/combinedcs.hi.train.tok

# Vocab File

CONCAT_BPE=$DATA_PATH/all

SRC_VOCAB=$DATA_PATH/vocab.hi
TGT_VOCAB=$DATA_PATH/vocab.en
FULL_VOCAB=$DATA_PATH/vocab.all

# Hi and En/CS test and valid files

SRC_VALID1=$MONO_PATH1/hi.valid
SRC_VALID_TOK1=$MONO_PATH1/hi.valid.tok

SRC_VALID2=$MONO_PATH2/hi.valid
SRC_VALID_TOK2=$MONO_PATH2/hi.valid.tok

SRC_VALID3=$MONO_PATH3/hi.valid
SRC_VALID_TOK3=$MONO_PATH3/hi.valid.tok

SRC_VALID4=$MONO_PATH4/combinedcs.hi.valid
SRC_VALID_TOK4=$MONO_PATH4/combinedcs.hi.valid.tok

TGT_VALID1=$MONO_PATH1/en.valid
TGT_VALID_TOK1=$MONO_PATH1/en.valid.tok

TGT_VALID2=$MONO_PATH2/en.valid
TGT_VALID_TOK2=$MONO_PATH2/en.valid.tok

TGT_VALID3=$MONO_PATH3/en.valid
TGT_VALID_TOK3=$MONO_PATH3/en.valid.tok

TGT_VALID4=$MONO_PATH4/combinedcs.en.valid
TGT_VALID_TOK4=$MONO_PATH4/combinedcs.en.valid.tok

SRC_TEST1=$MONO_PATH1/hi.test
SRC_TEST_TOK1=$MONO_PATH1/hi.test.tok

SRC_TEST2=$MONO_PATH2/hi.test
SRC_TEST_TOK2=$MONO_PATH2/hi.test.tok

SRC_TEST3=$MONO_PATH3/hi.test
SRC_TEST_TOK3=$MONO_PATH3/hi.test.tok

SRC_TEST4=$MONO_PATH4/combinedcs.hi.test
SRC_TEST_TOK4=$MONO_PATH4/combinedcs.hi.test.tok

TGT_TEST1=$MONO_PATH1/en.test
TGT_TEST_TOK1=$MONO_PATH1/en.test.tok

TGT_TEST2=$MONO_PATH2/en.test
TGT_TEST_TOK2=$MONO_PATH2/en.test.tok

TGT_TEST3=$MONO_PATH3/en.test
TGT_TEST_TOK3=$MONO_PATH3/en.test.tok

TGT_TEST4=$MONO_PATH4/combinedcs.en.test
TGT_TEST_TOK4=$MONO_PATH4/combinedcs.en.test.tok


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

echo "Extracting vocabulary..."
cat $SRC_RAW1 $SRC_RAW2 $SRC_RAW3 $SRC_RAW4 | $FASTBPE getvocab - >$SRC_VOCAB
cat $TGT_RAW1 $TGT_RAW2 $TGT_RAW3 $TGT_RAW4 | $FASTBPE getvocab - >$TGT_VOCAB
cat $SRC_RAW1 $SRC_RAW2 $SRC_RAW3 $SRC_RAW4 $TGT_RAW1 $TGT_RAW2 $TGT_RAW3 $TGT_RAW4 | $FASTBPE getvocab - >$FULL_VOCAB

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
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW3
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW4

python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW3
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW4

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

echo "Binarizing data..."
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID3
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID4

python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID3
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID4

python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST3
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST4

python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST1
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST2
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST3
python3 $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST4

# echo "Concatenating source and target monolingual data..."
cat $SRC_RAW1 $SRC_RAW2 $SRC_RAW3 $SRC_RAW4 $TGT_RAW1 $TGT_RAW2 $TGT_RAW3 $TGT_RAW4 | shuf > $CONCAT_BPE
# # fi
# echo "Concatenated data in: $CONCAT_BPE"

# # if ! [[ -f "$CONCAT_BPE.vec" ]]; then
echo "Training fastText on $CONCAT_BPE..."
$FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 256 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE".256"
# $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE".512"
# # fi
# # echo "Cross-lingual embeddings in: $CONCAT_BPE.vec"


