import unittest

import torch

from vlmeval.probes.generation_attention import Bounds, allowed_no_write, image_bounds, paged_predicate


class GenerationMaskTests(unittest.TestCase):
    def test_delimiters_and_prefix(self):
        ids = [11,12,151652,151655,151655,151653,21,22,
               151652,151655,151655,151653,21,22,31]
        b = image_bounds(ids)
        self.assertEqual(b,Bounds(2,8,12))
        q = torch.arange(len(ids))[:,None]
        k = torch.arange(len(ids))[None,:]
        actual = allowed_no_write(q,k,b)
        expected = torch.tensor([[j<=i and not (8<=i<12 and 2<=j<8)
                                  for j in range(len(ids))] for i in range(len(ids))])
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(int(((q>=k)&~actual).sum()),24)
        self.assertTrue(actual[8:12,:2].all())  # system/header prefix kept
        self.assertTrue(actual[12:,:12].all())  # T2 and answer still read V2
        self.assertTrue(actual[8:12,8:12].diag().all())

    def test_invalid_image_layout_rejected(self):
        for ids in [[1,2,3],[151655,151655,2,151655],[151652,151655,3,151652,151655,151653]]:
            with self.assertRaises((ValueError,IndexError)):
                image_bounds(ids)

    def paged_case(self,device='cpu',decode=False):
        from vllm.v1.attention.backends.flex_attention import FlexAttentionMetadata,causal_mask_mod
        m = object.__new__(FlexAttentionMetadata)
        m.block_size=8
        # Physical pages deliberately interleaved, not request-contiguous.
        m.physical_to_logical=torch.tensor([[-1,1,-1,0], [1,-1,0,-1]],device=device)
        m.seq_lens=torch.tensor([15,13],device=device)
        lengths=[2,1] if decode else [15,13]
        m.query_start_loc=torch.tensor([0,lengths[0],sum(lengths)],device=device)
        m.doc_ids=torch.tensor([0]*lengths[0]+[1]*lengths[1],device=device)
        m.decode_offset=torch.tensor([13,12] if decode else [0,0],device=device)
        m.logical_mask_mod=causal_mask_mod
        ranges=torch.tensor([[2,7,11,1],[1,4,8,1]],device=device,dtype=torch.int32)
        native=m.get_paged_mask_mod()
        mask=paged_predicate(m,native,ranges)
        expected=torch.zeros(sum(lengths),32,dtype=torch.bool,device=device)
        pages=[[3,1],[2,0]]
        for q in range(sum(lengths)):
            req=0 if q<lengths[0] else 1
            lq=q-(0 if req==0 else lengths[0])+([13,12][req] if decode else 0)
            lo,start,end,_=ranges[req].tolist()
            for k in range(32):
                page,off=divmod(k,8)
                if page not in pages[req]:continue
                lk=pages[req].index(page)*8+off
                expected[q,k]=(lk<int(m.seq_lens[req]) and lk<=lq
                               and not(start<=lq<end and lo<=lk<start))
        return mask,expected

    def test_paged_request_mapping_and_decode(self):
        for decode in [False,True]:
            mask,expected=self.paged_case(decode=decode)
            q=torch.arange(expected.shape[0])[:,None]
            k=torch.arange(32)[None,:]
            actual=mask(torch.tensor(0),torch.tensor(0),q,k)
            self.assertTrue(torch.equal(actual,expected))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_compiled_flex_matches_dense_sdpa(self):
        from torch.nn.attention.flex_attention import BlockMask, flex_attention
        mask,expected=self.paged_case(device='cuda')
        n=expected.shape[0]
        # A single partial block: no "full block" bypass of the predicate.
        bm=BlockMask.from_kv_blocks(torch.ones((1,1,1),device='cuda',dtype=torch.int32),
            torch.zeros((1,1,1,1),device='cuda',dtype=torch.int32),BLOCK_SIZE=(128,128),
            mask_mod=mask,seq_lengths=(n,32))
        torch.manual_seed(23)
        q=torch.randn(1,4,n,32,device='cuda')
        k=torch.randn(1,4,32,32,device='cuda');v=torch.randn_like(k)
        fn=torch.compile(flex_attention)
        actual=fn(q,k,v,block_mask=bm)
        reference=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=expected)
        torch.testing.assert_close(actual,reference,atol=2e-5,rtol=2e-5)
        # Perturb ONLY forbidden values: a blocked row's output must be invariant.
        changed=v.clone();changed[:,:,24+2:24+7]+=100
        altered=fn(q,k,changed,block_mask=bm)
        torch.testing.assert_close(actual[:,:,7:11],altered[:,:,7:11],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
