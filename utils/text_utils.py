import stanza


nlps = {}
tokenizers = {}


class TextUtils:
    """ This class contains functions and definitions related to text used across all the project. """

    @staticmethod
    def get_nlp(language):
        global nlps
        if language not in nlps:
            if language == 'English':
                nlps[language] = stanza.Pipeline('en', tokenize_no_ssplit=True)
            elif language == 'German':
                nlps[language] = stanza.Pipeline('de', tokenize_no_ssplit=True)
            elif language == 'Japanese':
                nlps[language] = stanza.Pipeline('ja', tokenize_no_ssplit=True)
            elif language == 'Chinese':
                nlps[language] = stanza.Pipeline('zh', tokenize_no_ssplit=True)
            elif language == 'French':
                nlps[language] = stanza.Pipeline('fr', tokenize_no_ssplit=True)

        return nlps[language]

    @staticmethod
    def get_tokenizer(language):
        global tokenizers
        if language not in tokenizers:
            if language == 'English':
                tokenizers[language] = stanza.Pipeline('en', processors='tokenize', tokenize_no_ssplit=True)
            elif language == 'German':
                tokenizers[language] = stanza.Pipeline('de', processors='tokenize', tokenize_no_ssplit=True)
            elif language == 'Japanese':
                tokenizers[language] = stanza.Pipeline('de', processors='tokenize', tokenize_no_ssplit=True)
            elif language == 'Chinese':
                tokenizers[language] = stanza.Pipeline('zh', processors='tokenize', tokenize_no_ssplit=True)
            elif language == 'French':
                tokenizers[language] = stanza.Pipeline('ja', processors='tokenize', tokenize_no_ssplit=True)

        return tokenizers[language]

    """ Given an analyzed sentence, check if its main verb is transitive. This is done by searching if there's a
        token which is a direct object of the root.
    """

    @staticmethod
    def transitive_exception(analyzed_sentence, language):
        """ Check for exceptions which seems transitive but are not. """
        if language == 'Chinese':
            root_ind = [i for i in range(len(analyzed_sentence)) if analyzed_sentence[i]['dep'].lower() == 'root'][0]
            return \
                analyzed_sentence[root_ind + 1]['lemma'] == '在' or \
                analyzed_sentence[root_ind + 1]['lemma'].startswith('向') or \
                analyzed_sentence[root_ind]['lemma'].endswith('在')
        if language == 'German':
            return len([
                token for token in analyzed_sentence
                if analyzed_sentence[token['head_ind']]['dep'].lower() == 'root' and
                token['xpos'] == 'PTKVZ'
            ]) > 0

    @staticmethod
    def is_transitive_sentence(analyzed_sentence, language):
        direct_object_dep_tag = 'obj'

        return len([
            token for token in analyzed_sentence
            if token['dep'] == direct_object_dep_tag
            and analyzed_sentence[token['head_ind']]['dep'].lower() == 'root'
            # We manually fix a Stanza bug
            and (not TextUtils.transitive_exception(analyzed_sentence, language))
        ]) > 0

    @staticmethod
    def is_existential_sentence(analyzed_sentence, language):
        """ Check if the sentence is existential ("There is a..."), assuming that there's a single root. """
        if language == 'English':
            be_verb = 'be'
        elif language == 'German':
            be_verb = 'sein'
        elif language == 'Chinese':
            be_verb = '有'
        elif language == 'Japanese':
            be_verb = 'ある'

        if language == 'Japanese':
            # In Japanese, if the be verb is in the end of the sentence, this is an existential sentence
            return analyzed_sentence[-1]['lemma'] == be_verb
        else:
            # In other languages, check if the root is the be verb
            root = [token for token in analyzed_sentence if token['dep'].lower() == 'root'][0]
            return root['lemma'].lower() == be_verb

    @staticmethod
    def prepare_caption_for_stanza(caption, language):
        if language in ['Chinese', 'Japanese']:
            dot_str = '。'
        else:
            dot_str = '.'
        return caption.replace('\n', dot_str + ' ')

    @staticmethod
    def tokenize(sentence, language):
        """ Tokenize an input sentence. """
        sentence = TextUtils.prepare_caption_for_stanza(sentence, language)
        return [x.to_dict()[0]['text'] for x in TextUtils.get_tokenizer(language)(sentence.lower()).sentences[0].tokens]

    @staticmethod
    def preprocess_token(token):
        token = "".join(c for c in token if c not in ("?", ".", ";", ":", "!", "'", "`", "\""))
        token = token.lower()

        return token

    @staticmethod
    def tokenize_and_clean(sentence, language):
        tokenized_sentence = TextUtils.tokenize(sentence, language)
        return [TextUtils.preprocess_token(x) for x in tokenized_sentence]

    @staticmethod
    def phrase_in_sent(tokenized_sentence, tokenized_phrase):
        for sent_ext_ind in range(len(tokenized_sentence)):
            phrase_ind = 0
            sent_int_ind = sent_ext_ind
            while phrase_ind < len(tokenized_phrase) and sent_int_ind < len(tokenized_sentence) and\
                    tokenized_sentence[sent_int_ind] == tokenized_phrase[phrase_ind]:
                phrase_ind += 1
                sent_int_ind += 1
            if phrase_ind == len(tokenized_phrase):
                return True
        return False

    @staticmethod
    def extract_nlp_info(analyzed_sentence):
        agg_token_lists = [[x.to_dict() for x in y.tokens] for y in analyzed_sentence.sentences]
        token_lists = []
        for agg_token_list in agg_token_lists:
            token_lists.append([])
            for agg_token in agg_token_list:
                if len(agg_token) == 1:
                    token_lists[-1].append(agg_token[0])
                else:
                    token_lists[-1] += agg_token[1:]
        return [[{
            'pos': x['upos'],
            'xpos': x['xpos'] if 'xpos' in x else x['upos'],
            'dep': x['deprel'],
            'lemma': x['lemma'] if 'lemma' in x else x['text'],
            'head_ind': x['head'] - 1
        } for x in y] for y in token_lists]
