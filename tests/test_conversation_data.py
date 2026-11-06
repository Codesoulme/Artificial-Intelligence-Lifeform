import unittest
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders
from scripts.prepare_conversation import SPECIAL,split_of,clean,conversation,encode_dialogue
from scripts.curate_conversation import reject

class ConversationDataTests(unittest.TestCase):
    def test_grouping_filters_and_loss_masks(self):
        self.assertEqual(split_of('相同 问题'),split_of('相同问题'))
        self.assertIsNone(clean('我是由别人开发的模型。'))
        self.assertIsNone(clean('<think>隐藏过程</think>答复。'))
        self.assertIsNone(conversation({'conversations':[{'role':'user','content':'请查天气。'},{'role':'assistant','content':'正在查询。','tool_calls':[{}]}]}))
        messages=[{'role':'user','content':'你好。'},{'role':'assistant','content':'你好！有什么需要帮助的？'},
                  {'role':'user','content':'请说再见。'},{'role':'assistant','content':'再见。'}]
        tokenizer=Tokenizer(models.BPE());tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder=decoders.ByteLevel()
        tokenizer.train_from_iterator([m['content'] for m in messages],trainers.BpeTrainer(vocab_size=300,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        for text in ('你好。','冷僻字龘和emoji🌱'):
            self.assertEqual(tokenizer.decode(tokenizer.encode(text).ids),text)
        tokens,mask=encode_dialogue(tokenizer,messages)
        active=False
        for token,label in zip(tokens,mask):
            if token==tokenizer.token_to_id('<|assistant|>'):active=True;self.assertFalse(label)
            elif token in (tokenizer.token_to_id('<|user|>'),tokenizer.token_to_id('<|bos|>')):active=False;self.assertFalse(label)
            else:self.assertEqual(label,active)
        self.assertIsNone(encode_dialogue(tokenizer,messages,limit=8))

    def test_second_pass_rejects_known_identity_and_experience_leaks(self):
        self.assertEqual(reject({'text':'我名为MiniMind，由jingyaogong开发。'},'pretrain'),'identity')
        self.assertEqual(reject({'text':'你是否基于特定数据集训练？'},'pretrain'),'identity')
        row={'text':'最近忙什么？','messages':[{'role':'user','content':'最近忙什么？'},{'role':'assistant','content':'最近和朋友一起做饭。'}]}
        self.assertEqual(reject(row,'dialogue'),'invented_experience')
        self.assertIsNone(reject({'text':'小兔子在森林中迷路，后来找到了回家的路。'},'pretrain'))

if __name__=='__main__':unittest.main()
