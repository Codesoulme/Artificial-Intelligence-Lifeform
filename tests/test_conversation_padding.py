import copy,unittest,torch
from spora.temporal import TemporalSPORA,TemporalConfig
from spora.growing import new_state
from spora.conversation import ConversationBridge,ConversationConfig
from scripts.train_conversation_v4 import padded,batch_loss

class PaddingTests(unittest.TestCase):
    def test_short_right_padding_keeps_supervised_loss_and_gradients(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        core=TemporalSPORA(TemporalConfig(width=32,layers=2,heads=2,memory_width=64,expert_rank=4,memory_layers=(0,1),history_writer='rls',current_address='orthogonal'))
        state=new_state(core);model=ConversationBridge(core,ConversationConfig(vocabulary=64,width=32,depth=2,heads=4,hidden=64,context=64,memory_layers=(0,1)))
        other=copy.deepcopy(model)
        rows=[dict(tokens=[1,3,15,17,2,4,21,22,2],mask=[False]*6+[True]*3),dict(tokens=[1,3,8,2,4,31,2],mask=[False]*5+[True]*2)]
        x,y=padded(rows,16,0,'cpu');a=batch_loss(model,state,x,y);a.backward()
        x,y=padded(rows,32,0,'cpu');b=batch_loss(other,state,x,y);b.backward()
        self.assertAlmostEqual(float(a),float(b),places=6)
        for p,q in zip(model.trainable_parameters(),other.trainable_parameters()):
            self.assertTrue(torch.allclose(p.grad,q.grad,atol=2e-6,rtol=2e-5))

if __name__=='__main__':unittest.main()
