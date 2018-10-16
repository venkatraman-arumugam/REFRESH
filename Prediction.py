import tensorflow as tf

from model_utils import convert_logits_to_softmax
from my_model import MY_Model
from my_flags import FLAGS


class Summarizer:
    def __init__(self, vocab_dict, word_embedding_array):
        config = tf.ConfigProto(allow_soft_placement=True)
        self.session = tf.Session(config = config)

        self.model = MY_Model(self.session, len(vocab_dict) - 2)

        # Reload saved model and test
        print("Reading model parameters")
        self.model.saver.restore(self.session,FLAGS.model_dir + '/model.ckpt.epoch-' + str(FLAGS.model_to_load))
        print("Model loaded.")

        # Initialize word embedding before training
        print("Initialize word embedding vocabulary with pretrained embeddings ...")
        self.session.run(self.model.vocab_embed_variable.assign(word_embedding_array))

    def prediction(self, batch_document_modelIn, doclength):
        batch_logits = self.session.run(self.model.logits,
                                        feed_dict={self.model.document_placeholder: batch_document_modelIn})
        batch_softmax_logits = convert_logits_to_softmax(batch_logits, session=self.session)

        print(batch_softmax_logits)

        softmax_logits = batch_softmax_logits[0]
        # Find top three scoring sentence to consider for summary, if score is same, select sentences with low indices
        oneprob_sentidx = {}
        for sentidx in range(FLAGS.max_doc_length):
            prob = softmax_logits[sentidx][0]  # probability of predicting one
            if sentidx < doclength:
                if prob not in oneprob_sentidx:
                    oneprob_sentidx[prob] = [sentidx]
                else:
                    oneprob_sentidx[prob].append(sentidx)
            else:
                break
        oneprob_keys = oneprob_sentidx.keys()
        sorted(oneprob_keys, reverse=True)

        # Rank sentences with scores: if same score lower ones ranked first
        sentindices = []
        for oneprob in oneprob_keys:
            sent_withsamescore = oneprob_sentidx[oneprob]
            sent_withsamescore.sort()
            sentindices += sent_withsamescore

        # Select Top 3
        print(sentindices)
        final_sentences = sentindices[:3]
        return final_sentences