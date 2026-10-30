import numpy as np
from backend import joint_context

def test_joint_encoder_outputs_three_bounded_heads_and_empty_input():
 gray=np.full((100,100),255,np.uint8);gray[45:55,10:90]=0;ink=255-gray
 points=[{'pixel':[45,50],'side':'right'},{'pixel':[50,50],'kind':'tee'}]
 result=joint_context.predict(gray,ink,points)
 for key,size in [('port',2),('junction',4),('line',3)]:
  assert result[key].shape==(2,size) and np.isfinite(result[key]).all()
  assert (result[key]>=0).all() and np.allclose(result[key].sum(1),1)
 assert joint_context.predict(gray,ink,[])['port'].shape==(0,2)
