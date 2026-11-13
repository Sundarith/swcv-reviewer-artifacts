import unittest
from contracts import action,evidence_pool,strips,select_strips
class Contracts(unittest.TestCase):
 def test_branches(self):
  self.assertEqual(action([.59,-2]),'correct');self.assertEqual(action([-1,-1.2]),'incorrect');self.assertEqual(action([-.99,-1.2]),'ambiguous')
 def test_branches_keep_pool_local_and_deduplicate(self):
  self.assertEqual(evidence_pool('correct',[1,2],[2,3]),[1,2]);self.assertEqual(evidence_pool('incorrect',[1,2],[2,3]),[2,3]);self.assertEqual(evidence_pool('ambiguous',[1,2],[2,3]),[1,2,3])
 def test_no_irrelevant_strip_fallback(self):self.assertEqual(select_strips([dict(score=-.6,text='bad')]),[])
 def test_threshold_ties_and_limit(self):
  items=[dict(score=.1,index=i) for i in range(7)];self.assertEqual([x['index'] for x in select_strips(items)],list(range(5)));self.assertEqual(select_strips([dict(score=-.5)]),[])
 def test_sentence_content_preserved(self):
  self.assertEqual(strips('One. Two? Three! Four.'),['One. Two? Three!','Four.'])
if __name__=='__main__':unittest.main()
