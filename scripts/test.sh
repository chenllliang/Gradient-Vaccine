root_dir=../data
lang_list=$root_dir/lang_list.txt  # <path to a file which contains a list of languages separted by new lines>
lang_pairs="fr_XX-en_XX,ro_RO-en_XX"

path_2_data=$1
model=$2
out_dir=$2-test_output

source_lang=(ro_RO fr_XX)
TGT=(en_XX)
export CUDA_VISIBLE_DEVICES=0

for i in $(seq 0 1); do
for j in $(seq 0 0); do
if [ ${source_lang[$i]} != ${TGT[$j]} ]
then

echo testing ${source_lang[$i]} - ${TGT[$j]}

mkdir $out_dir

fairseq-generate "$path_2_data" \
  --path $model \
  --task translation_multi_simple_epoch \
  --gen-subset test \
  --source-lang "${source_lang[$i]}" \
  --target-lang "${TGT[$j]}" \
  --batch-size 100 --remove-bpe 'sentencepiece'\
  --encoder-langtok "src" \
  --beam 5 \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" > $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5


cat $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5 | grep -P "^H"  |cut -f 3-  > $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.hyp
cat $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5 | grep -P "^T"  |cut -f 2-  > $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.ref

echo sacrebleu ${TGT[$j]} 
sacrebleu $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.ref < $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.hyp -w 2
sacrebleu $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.ref < $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.hyp -w 2 > $out_dir/${source_lang[$i]}-${TGT[$j]}.beam5.sacrebleu

fi
done
done