DICT=shared_dict_path
DEST=../data/multilingual_bin
DATA=sentencepieced_data_path

mkdir $DEST

TGT=en_XX
SRC=(fr_XX ro_RO)

for i in $(seq 0 1); do

NAME=$SRC-${TGT[$i]}

echo preprocessing $NAME

fairseq-preprocess \
  --source-lang ${SRC[$i]} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/${TRAIN}/train.spm.${SRC[$i]}-${TGT} \
  --validpref ${DATA}/${VALID}/valid.spm.${SRC[$i]}-${TGT} \
  --testpref ${DATA}/${TEST}/test.spm.${SRC[$i]}-${TGT} \
  --destdir ${DEST} \
  --joined-dictionary \
  --tgtdict $DICT \
  --workers 50 \

done