
# Global objects
import datetime
import hashlib
import subprocess
import time
import nltk
from Prediction import Summarizer
from data_utils import DataProcessor

PAD_ID = 0
UNK_ID = 1
vocab_dict, word_embedding_array = DataProcessor().prepare_vocab_embeddingdict()
# # print (len(vocab_embed_object.vocab_dict)-2)
model_cpu = Summarizer(vocab_dict, word_embedding_array)

class Preprocess:

    def timestamp(self):
        return datetime.datetime.fromtimestamp(time.time()).strftime('[%Y-%m-%d %H:%M:%S]')

    def Hashhex(self, s):
        """Returns a heximal formated SHA1 hash of the input string.
        Args:
          s: The string to hash.
        Returns:
          A heximal formatted hash of the input string.
        """
        h = hashlib.sha1()
        h.update(s)
        return h.hexdigest()

    def stanford_processing(self, log, story, highlights):
        story_corenlp = None
        highlights_corenlp = None
        try:
            log += self.timestamp() + " Start Stanford Processing (SSegmentation,Tokenization,NERTagging) ...\n"

            story_corenlp = subprocess.check_output(['./corenlp.sh', story])
            highlights_corenlp = subprocess.check_output(['./corenlp.sh', highlights])

            log += self.timestamp() + " Stanford Processing finished.\n"
        except Exception as e:
            log += self.timestamp() + " Stanford Processing failed.\n" + str(e) + "\n"

        return log, story_corenlp, highlights_corenlp

    def corenlp_output_parser(self, text):

        data_org = []
        # data_ner = []
        # data_orglower_anonym = []
        data_org_vocabid = []

        # Parse Stanford Output Data
        # sentdata_list = corenlp_output.strip().split("Sentence #")[1:]
        for sentdata in nltk.sent_tokenize(text):
            line_org = []
            # line_ner = []
            for word in nltk.word_tokenize(sentdata):
                line_org.append(word)
                        # if token.startswith("NamedEntityTag="):
                        #     if token.startswith("NamedEntityTag=PERSON"):
                        #         line_ner.append("PERSON")
                        #     elif token.startswith("NamedEntityTag=LOCATION"):
                        #         line_ner.append("LOCATION")
                        #     elif token.startswith("NamedEntityTag=ORGANIZATION"):
                        #         line_ner.append("ORGANIZATION")
                        #     elif token.startswith("NamedEntityTag=MISC"):
                        #         line_ner.append("MISC")
                        #     else:
                        #         line_ner.append("O")
            data_org.append(line_org)
            # data_ner.append(line_ner)

            line_org_vocabid = [vocab_dict[word] if word in vocab_dict else UNK_ID
                                for word in line_org]
            data_org_vocabid.append(line_org_vocabid)

        return data_org, data_org_vocabid  # data_ner, data_orglower_anonym

    def stanford_output_modelIn_processing(self, log, story_corenlp, highlights_corenlp):
        story_line_org = None
        highlights_line_org = None
        document_modelIn = None

        try:
            log += self.timestamp() + " Start model input preparation (StanOutputParsing,OriginalCases,NotAnonymized,VocabIdMap) ...\n"

            story_line_org, story_org_vocabid = self.corenlp_output_parser(story_corenlp)
            # print story_line_org, story_orglower_anonym_vocabid

            highlights_line_org, _ = self.corenlp_output_parser(highlights_corenlp)
            # print highlights_line_org

            document_modelIn = DataProcessor().prepare_document_modelIn(story_org_vocabid, [], [])
            # print document_modelIn

            log += self.timestamp() + " Model input preparation finished.\n"
        except Exception as e:
            log += self.timestamp() + " Model input preparation failed.\n" + str(e) + "\n"

        # print story_line_org, highlights_line_org, document_modelIn
        # print document_modelIn.shape

        return log, story_line_org, highlights_line_org, document_modelIn

    def refresh_prediction(self, log, document_modelIn, doclen):
        # global model_cpu

        # print document_modelIn, doclen
        selected_sentids = None
        try:
            log += self.timestamp() + " Start predicting with Refresh (Best CNN-trained model from Narayan, Cohen and Lapata, 2018) ...\n"

            selected_sentids = model_cpu.prediction(document_modelIn, doclen)

            log += self.timestamp() + " Refresh prediction finished.\n"
        except Exception as e:
            log += self.timestamp() + " Refresh prediction failed.\n" + str(e) + "\n"

        return log, selected_sentids

    def run_textmode(self, text):
        '''Text MODE
        '''
        # Start a log
        log = ""

        try:
            log += self.timestamp() + " Summarizing a text: No side information used.\n"

            # No HTML Parsing and Text Extraction Needed
            story = text
            highlights = ""

            # # Start Stanford Parsing for Sentence Segmentation, Tokenization and NER Tagging
            # log, story_corenlp, highlights_corenlp = self.stanford_processing(log, story, highlights)
            # print(log)
            # if (story_corenlp is None) or (highlights_corenlp is None):
            #     raise Exception
            # print story_corenlp, highlights_corenlp

            # Stanford Output Parsing and Preparing input to the model
            log, story_line_org, highlights_line_org, document_modelIn = self.stanford_output_modelIn_processing(log,
                                                                                                            story,
                                                                                                                 highlights)
            print(log)
            if (story_line_org is None) or (highlights_line_org is None) or (document_modelIn is None):
                raise Exception
            # print story_line_org, highlights_line_org, document_modelIn
            # print document_modelIn.shape

            # SideNet Prediction
            log, selected_sentids = self.refresh_prediction(log, document_modelIn, len(story_line_org))
            print(log)
            if (selected_sentids is None):
                raise Exception
            selected_sentids.sort()
            print(selected_sentids)

            # Generate final outputs
            log += self.timestamp() + " Producing output summaries. \n"
            slead = "\n".join([" ".join(sent) for sent in story_line_org[:3]])
            srefresh = "\n".join([" ".join(story_line_org[sidx]) for sidx in selected_sentids])
            sgold = "\n".join([" ".join(sent) for sent in highlights_line_org])

            # print log
            # print slead
            # print ssidenet
            # print sgold

            return log, slead, srefresh, sgold

        except Exception as e:
            log += self.timestamp() + " Failed.\n" + str(e) + "\n"
            print(log)
            return log, "", "", ""
