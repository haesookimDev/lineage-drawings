import numpy as np
from PIL import Image,ImageDraw
from backend.line_recovery import periodic,recover,dashed_arms,profile,short_links
from backend.topology_context import prepare
from backend.sparse_topology import intersections

def test_regular_dashes_bridge_beyond_old_ten_pixel_gap():
 im=Image.new('RGB',(240,160),'white');d=ImageDraw.Draw(im)
 for x in range(20,200,24):d.line((x,80,x+9,80),fill='black',width=2)
 _,ink=prepare(im);lines,n=recover(ink,[])
 assert n and any(a<25 and c>190 and abs(b-80)<2 and abs(z-80)<2 for a,b,c,z in lines)
 assert periodic(np.tile(np.r_[np.ones(10,bool),np.zeros(14,bool)],6))


def test_irregular_text_like_gaps_are_not_a_dashed_line():
 hit=np.zeros(120,bool)
 for a,b in [(0,4),(6,10),(15,20),(46,50),(91,96)]:hit[a:b]=True
 assert not periodic(hit)


def test_dashed_fourth_arm_cannot_turn_plain_cross_into_tee():
 im=Image.new('RGB',(220,220),'white');d=ImageDraw.Draw(im);d.line((20,110,200,110),fill='black',width=2);d.line((110,20,110,110),fill='black',width=2)
 for y in range(119,210,21):d.line((110,y,110,y+7),fill='black',width=2)
 _,ink=prepare(im);rows=intersections([[20,110,200,110],[110,20,110,200]],[[],[]],ink,recover=True)
 assert rows and rows[0]['arms']==4


def test_batched_dashed_arms_match_scalar_pixel_reference():
 rng=np.random.default_rng(24);ink=(rng.random((170,220))>.7).astype(np.uint8)*255;points=rng.uniform([0,0],[219,169],(50,2))
 expected=np.array([[periodic(profile(ink,x+dx*3,y+dy*3,x+dx*65,y+dy*65)) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))] for x,y in points])
 assert np.array_equal(dashed_arms(ink,points),expected)


def test_two_dash_signal_between_faces_needs_real_ink_not_empty_gap():
 els=[{'id':'a','kind':'unknown','bbox':[.1,.2,.3,.4]},{'id':'b','kind':'unknown','bbox':[.7,.2,.9,.4]}]
 ink=np.zeros((100,100),np.uint8);ink[30,30:42]=255;ink[30,56:71]=255
 assert len(short_links(els,ink,100,100))==1
 assert not short_links(els,np.zeros_like(ink),100,100)
 assert not short_links([els[0],{**els[1],'bbox':[.2,.2,.4,.4]}],ink,100,100)
