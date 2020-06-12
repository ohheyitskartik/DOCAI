from docproduct.predictor import RetreiveQADoc

pretrained_path = 'BioBertFolder/biobert_v1.0_pubmed_pmc/'
# ffn_weight_file = None
bert_ffn_weight_file = 'newFolder/models/bertffn_crossentropy/bertffn'
embedding_file = 'Float16EmbeddingsExpanded5-27-19.pkl'

doc = RetreiveQADoc(pretrained_path=pretrained_path,
                    ffn_weight_file=None,
                    bert_ffn_weight_file=bert_ffn_weight_file,
                    embedding_file=embedding_file)


question_text = "i am 18 and having frequent and painful urination, with a mild fever since 2-3 days"

search_similarity_by = 'answer'

number_results_toReturn = 10

answer_only = True

returned_results = doc.predict(question_text,
                               search_by=search_similarity_by, topk=number_results_toReturn, answer_only=answer_only)

print('')
for jk in range(len(returned_results)):
    print("Result ", jk+1)
    print(returned_results[jk])
    print('')
