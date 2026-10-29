"""Weak branch context must not inherit strong endpoint confidence."""
from backend.geometry import combine_context_pairs


def test_uncertain_traversed_junction_does_not_create_supplemental_edge():
    elements=[{'id':k,'kind':'unknown','source':'neural_symbol_candidate',
               'properties':{'detector_score':.9}} for k in ('a','b','c')]
    paths=[{'from_id':'a','to_id':k,'score':.95,'junction_score':quality,
            'points':[[0,0],[0,.5],[1,.5]]} for k,quality in [('b',.4),('c',.9)]]
    result,_=combine_context_pairs(elements,[],paths,{'extra_junction_threshold':.55})
    assert result==[('a','c')]
    # A supplemental quality gate cannot erase an existing validated edge.
    retained,_=combine_context_pairs(elements,[('a','b')],paths,{'extra_junction_threshold':.55})
    assert set(retained)=={('a','b'),('a','c')}
