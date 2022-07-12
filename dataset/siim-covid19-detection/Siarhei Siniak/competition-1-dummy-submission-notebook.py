import pandas as pd
from path import Path
import pprint

def display(*args, **kwargs):
    pprint.pprint([args, kwargs])
    
dataset_path = Path('../input/siim-covid19-detection')
submission_df = pd.read_csv(dataset_path/'sample_submission.csv')
display(submission_df.head())
display(submission_df.iloc[2000:2010])
display(submission_df.to_csv('submission.csv', index=False))

#http://ix.io/3rxn
t2 = r'''
id,PredictionString
00188a671292_study,Negative 1 0 0 1 1
004bd59708be_study,Negative 1 0 0 1 1
00508faccd39_study,Negative 1 0 0 1 1
006486aa80b2_study,Negative 1 0 0 1 1
00655178fdfc_study,Negative 1 0 0 1 1
00a81e8f1051_study,Negative 1 0 0 1 1
00be7de16711_study,Negative 1 0 0 1 1
00c7a3928f0f_study,Negative 1 0 0 1 1
00d63957bc3a_study,Negative 1 0 0 1 1
0107f2d291d6_study,Negative 1 0 0 1 1
0154653179fa_study,Negative 1 0 0 1 1
015f89ec55ea_study,Negative 1 0 0 1 1
0241bc13eac6_study,Negative 1 0 0 1 1
025bfc117ff8_study,Negative 1 0 0 1 1
028abd3504b6_study,Negative 1 0 0 1 1
02ee3a9820eb_study,Negative 1 0 0 1 1
0321bb7f84b5_study,Negative 1 0 0 1 1
03e0a59d9b8a_study,Negative 1 0 0 1 1
03fc9ec0dba8_study,Negative 1 0 0 1 1
045783dbe7d1_study,Negative 1 0 0 1 1
04c42f33c006_study,Negative 1 0 0 1 1
04c4e92d40b8_study,Negative 1 0 0 1 1
04d0f88f790a_study,Negative 1 0 0 1 1
059007b73d82_study,Negative 1 0 0 1 1
05ce6249b5c0_study,Negative 1 0 0 1 1
05f22f0c2412_study,Negative 1 0 0 1 1
061969abb8a5_study,Negative 1 0 0 1 1
063ee1363644_study,Negative 1 0 0 1 1
06d315ac12ff_study,Negative 1 0 0 1 1
06fa49182629_study,Negative 1 0 0 1 1
070d030a9ff7_study,Negative 1 0 0 1 1
07467db0a5ae_study,Negative 1 0 0 1 1
074763143b55_study,Negative 1 0 0 1 1
07bac22f0a34_study,Negative 1 0 0 1 1
080ea6634d62_study,Negative 1 0 0 1 1
0816f9d3875e_study,Negative 1 0 0 1 1
0838c0a2c424_study,Negative 1 0 0 1 1
0843b8e9ca34_study,Negative 1 0 0 1 1
0847751da0f7_study,Negative 1 0 0 1 1
08cc9af3edf8_study,Negative 1 0 0 1 1
096601e1340d_study,Negative 1 0 0 1 1
098be79259d5_study,Negative 1 0 0 1 1
0992f862466e_study,Negative 1 0 0 1 1
0a50336ef885_study,Negative 1 0 0 1 1
0ac42f524362_study,Negative 1 0 0 1 1
0b1d3c1abed3_study,Negative 1 0 0 1 1
0b1f60821d8f_study,Negative 1 0 0 1 1
0b55f8263527_study,Negative 1 0 0 1 1
0b6af2d5fa06_study,Negative 1 0 0 1 1
0ba4b75a0784_study,Negative 1 0 0 1 1
0bc43d526b5d_study,Negative 1 0 0 1 1
0bdd39af9529_study,Negative 1 0 0 1 1
0cca5e135833_study,Negative 1 0 0 1 1
0d21581c34cf_study,Negative 1 0 0 1 1
0d476d070d71_study,Negative 1 0 0 1 1
0d7e69753505_study,Negative 1 0 0 1 1
0d90fdee21da_study,Negative 1 0 0 1 1
0dd6457640a2_study,Negative 1 0 0 1 1
0e5a61d4ecaf_study,Negative 1 0 0 1 1
0e6a7ad528a2_study,Negative 1 0 0 1 1
0e72153af29d_study,Negative 1 0 0 1 1
0e943735788f_study,Negative 1 0 0 1 1
0ebc8d73fad5_study,Negative 1 0 0 1 1
0ebcd444ab94_study,Negative 1 0 0 1 1
0ec0901a5d45_study,Negative 1 0 0 1 1
0ecd0d14d4f4_study,Negative 1 0 0 1 1
0f6a24630aae_study,Negative 1 0 0 1 1
0f78e4f5c662_study,Negative 1 0 0 1 1
0fa7372c4120_study,Negative 1 0 0 1 1
0fb12a40b091_study,Negative 1 0 0 1 1
102b201f8074_study,Negative 1 0 0 1 1
10450f9a0671_study,Negative 1 0 0 1 1
106326aea184_study,Negative 1 0 0 1 1
108d012276cd_study,Negative 1 0 0 1 1
10e136a7def6_study,Negative 1 0 0 1 1
10e407d29b98_study,Negative 1 0 0 1 1
112fca0c27c7_study,Negative 1 0 0 1 1
1150b14c8978_study,Negative 1 0 0 1 1
11808a6c4ebc_study,Negative 1 0 0 1 1
119e45ffb508_study,Negative 1 0 0 1 1
1204a99587e8_study,Negative 1 0 0 1 1
124796c05943_study,Negative 1 0 0 1 1
12d16bfd378c_study,Negative 1 0 0 1 1
12d71150e1c6_study,Negative 1 0 0 1 1
12e91c6be34e_study,Negative 1 0 0 1 1
134a29f5263d_study,Negative 1 0 0 1 1
13dd8cf5ae88_study,Negative 1 0 0 1 1
14072c7f2a75_study,Negative 1 0 0 1 1
1432f644f41a_study,Negative 1 0 0 1 1
1448dfb57548_study,Negative 1 0 0 1 1
147a09511a1b_study,Negative 1 0 0 1 1
148c96b41de4_study,Negative 1 0 0 1 1
149c8d66e874_study,Negative 1 0 0 1 1
14be2e12c7cf_study,Negative 1 0 0 1 1
1517e9b8a9f3_study,Negative 1 0 0 1 1
1595ac3e0da9_study,Negative 1 0 0 1 1
15c748a291eb_study,Negative 1 0 0 1 1
163abcbc5d4a_study,Negative 1 0 0 1 1
164b05333dd6_study,Negative 1 0 0 1 1
164de8fd99f3_study,Negative 1 0 0 1 1
1655b1e7933e_study,Negative 1 0 0 1 1
16924e50c674_study,Negative 1 0 0 1 1
17562fa44338_study,Negative 1 0 0 1 1
1777481e665f_study,Negative 1 0 0 1 1
17ffbc3d2207_study,Negative 1 0 0 1 1
1834c71c3ae8_study,Negative 1 0 0 1 1
1846f85eefa1_study,Negative 1 0 0 1 1
18acddb9abb7_study,Negative 1 0 0 1 1
18b75ba2b7e9_study,Negative 1 0 0 1 1
18c4af7e2309_study,Negative 1 0 0 1 1
18c748563dad_study,Negative 1 0 0 1 1
18d983a50a6c_study,Negative 1 0 0 1 1
18ffa70a4a16_study,Negative 1 0 0 1 1
191774101c9d_study,Negative 1 0 0 1 1
19a13254e857_study,Negative 1 0 0 1 1
19a57af80883_study,Negative 1 0 0 1 1
19b12e65a279_study,Negative 1 0 0 1 1
19b85f5e7b8c_study,Negative 1 0 0 1 1
19b9a8cb7587_study,Negative 1 0 0 1 1
19c66935e737_study,Negative 1 0 0 1 1
1a066eb95115_study,Negative 1 0 0 1 1
1a3ded50a43c_study,Negative 1 0 0 1 1
1a59ea3cecd6_study,Negative 1 0 0 1 1
1a77f709c788_study,Negative 1 0 0 1 1
1a9c5c2f7e59_study,Negative 1 0 0 1 1
1a9f44aee665_study,Negative 1 0 0 1 1
1aa3bb040531_study,Negative 1 0 0 1 1
1ae25872bc71_study,Negative 1 0 0 1 1
1ae711bf2282_study,Negative 1 0 0 1 1
1b11175dece3_study,Negative 1 0 0 1 1
1b196a72b78e_study,Negative 1 0 0 1 1
1b2461a133fe_study,Negative 1 0 0 1 1
1b318f080b44_study,Negative 1 0 0 1 1
1b59d0b0ce5e_study,Negative 1 0 0 1 1
1b7084d11c83_study,Negative 1 0 0 1 1
1b742fd5cbb8_study,Negative 1 0 0 1 1
1ba7f3f733ef_study,Negative 1 0 0 1 1
1bc28a510f3f_study,Negative 1 0 0 1 1
1be8a02fd932_study,Negative 1 0 0 1 1
1c57d288256c_study,Negative 1 0 0 1 1
1c68665f6172_study,Negative 1 0 0 1 1
1c716d133c0c_study,Negative 1 0 0 1 1
1c9e697113ad_study,Negative 1 0 0 1 1
1cc565a82e77_study,Negative 1 0 0 1 1
1d1ad3db8061_study,Negative 1 0 0 1 1
1d2ad8808b78_study,Negative 1 0 0 1 1
1d35f2fd66d3_study,Negative 1 0 0 1 1
1d3a22017cc8_study,Negative 1 0 0 1 1
1d4a182c0628_study,Negative 1 0 0 1 1
1d4d06e6949b_study,Negative 1 0 0 1 1
1d54c8a15099_study,Negative 1 0 0 1 1
1d7b78acc1ff_study,Negative 1 0 0 1 1
1d8b4a15135f_study,Negative 1 0 0 1 1
1dd2fb2d3f1e_study,Negative 1 0 0 1 1
1de50b53d3ed_study,Negative 1 0 0 1 1
1df9ecb29c1d_study,Negative 1 0 0 1 1
1e3602b92e0e_study,Negative 1 0 0 1 1
1e4038367823_study,Negative 1 0 0 1 1
1ea213ae57bd_study,Negative 1 0 0 1 1
1ebfe0b61513_study,Negative 1 0 0 1 1
1ecac8a297f4_study,Negative 1 0 0 1 1
1ee21b7aa650_study,Negative 1 0 0 1 1
1ee9fcb22c59_study,Negative 1 0 0 1 1
1f19df99fff0_study,Negative 1 0 0 1 1
1f816ad2d311_study,Negative 1 0 0 1 1
1facfeaad22f_study,Negative 1 0 0 1 1
1fc194fcbdb4_study,Negative 1 0 0 1 1
201cb653921c_study,Negative 1 0 0 1 1
2020aec5ec84_study,Negative 1 0 0 1 1
203e5f34c76c_study,Negative 1 0 0 1 1
204153b77776_study,Negative 1 0 0 1 1
2052c5acf159_study,Negative 1 0 0 1 1
2099b41c14a9_study,Negative 1 0 0 1 1
20d3b1fee58f_study,Negative 1 0 0 1 1
20d832153c0e_study,Negative 1 0 0 1 1
20ead06f3576_study,Negative 1 0 0 1 1
21119216b217_study,Negative 1 0 0 1 1
21367720ba0e_study,Negative 1 0 0 1 1
215d5a459a13_study,Negative 1 0 0 1 1
2169cfeca358_study,Negative 1 0 0 1 1
21c33e128420_study,Negative 1 0 0 1 1
21da4a896482_study,Negative 1 0 0 1 1
221ee7b67fc0_study,Negative 1 0 0 1 1
22c53dd77858_study,Negative 1 0 0 1 1
22d153d9e949_study,Negative 1 0 0 1 1
2326c314794c_study,Negative 1 0 0 1 1
234254cd1e73_study,Negative 1 0 0 1 1
23847f6a41a6_study,Negative 1 0 0 1 1
240db28d224f_study,Negative 1 0 0 1 1
241f7cd77858_study,Negative 1 0 0 1 1
24295591e074_study,Negative 1 0 0 1 1
243d12d1a64b_study,Negative 1 0 0 1 1
243e05e85c45_study,Negative 1 0 0 1 1
2478cdc439fa_study,Negative 1 0 0 1 1
250c0435c86c_study,Negative 1 0 0 1 1
2557cbfb0677_study,Negative 1 0 0 1 1
2558bb1ba493_study,Negative 1 0 0 1 1
2638265181c6_study,Negative 1 0 0 1 1
26c724e30c27_study,Negative 1 0 0 1 1
26f0f7350e8a_study,Negative 1 0 0 1 1
272c8b46d39d_study,Negative 1 0 0 1 1
27387026e831_study,Negative 1 0 0 1 1
2767827cedc2_study,Negative 1 0 0 1 1
279348dd03ec_study,Negative 1 0 0 1 1
2800a1cd98b3_study,Negative 1 0 0 1 1
28156ad1b4cc_study,Negative 1 0 0 1 1
283a0639fb59_study,Negative 1 0 0 1 1
286d4c0fe851_study,Negative 1 0 0 1 1
290e8b4dcae6_study,Negative 1 0 0 1 1
299e6030efab_study,Negative 1 0 0 1 1
29c547a00bbb_study,Negative 1 0 0 1 1
2a1a15f7fe7b_study,Negative 1 0 0 1 1
2a337009e493_study,Negative 1 0 0 1 1
2a72cf9ac79a_study,Negative 1 0 0 1 1
2a8a7f025b25_study,Negative 1 0 0 1 1
2ad7c6c0c11b_study,Negative 1 0 0 1 1
2aedd31c2d4f_study,Negative 1 0 0 1 1
2affce8b4145_study,Negative 1 0 0 1 1
2b0e4071bd4a_study,Negative 1 0 0 1 1
2b27117c51f8_study,Negative 1 0 0 1 1
2b48f1114500_study,Negative 1 0 0 1 1
2b53afb41e35_study,Negative 1 0 0 1 1
2b8321bcf4b9_study,Negative 1 0 0 1 1
2b8f261c9123_study,Negative 1 0 0 1 1
2b998af7ca8b_study,Negative 1 0 0 1 1
2b9ad8c6a010_study,Negative 1 0 0 1 1
2be4b695bf05_study,Negative 1 0 0 1 1
2c0138f0333f_study,Negative 1 0 0 1 1
2c12e56b6ef5_study,Negative 1 0 0 1 1
2c309f472123_study,Negative 1 0 0 1 1
2c4e544780bd_study,Negative 1 0 0 1 1
2c7ef4281f3e_study,Negative 1 0 0 1 1
2ca4713ccd6d_study,Negative 1 0 0 1 1
2cc5979920c6_study,Negative 1 0 0 1 1
2cd7576844db_study,Negative 1 0 0 1 1
2cfddd82d536_study,Negative 1 0 0 1 1
2d2e8475f384_study,Negative 1 0 0 1 1
2d5241612c2e_study,Negative 1 0 0 1 1
2d61c98874b9_study,Negative 1 0 0 1 1
2d673cbacf8e_study,Negative 1 0 0 1 1
2d9814778fe6_study,Negative 1 0 0 1 1
2de1b57b78a5_study,Negative 1 0 0 1 1
2e3ae25addcc_study,Negative 1 0 0 1 1
2e625aa9c04f_study,Negative 1 0 0 1 1
2ea214f65500_study,Negative 1 0 0 1 1
2eb19ca07152_study,Negative 1 0 0 1 1
2ebd6459c760_study,Negative 1 0 0 1 1
2ecc8e52fed2_study,Negative 1 0 0 1 1
2f26247d5fa4_study,Negative 1 0 0 1 1
2f45d8909a99_study,Negative 1 0 0 1 1
2f6093195a01_study,Negative 1 0 0 1 1
2f8d9f7afedb_study,Negative 1 0 0 1 1
2f92655213ca_study,Negative 1 0 0 1 1
2f9c9ca1ff98_study,Negative 1 0 0 1 1
2fb11712bc93_study,Negative 1 0 0 1 1
2fc50bf199cd_study,Negative 1 0 0 1 1
2ffd833491b2_study,Negative 1 0 0 1 1
302ff5bcbcf7_study,Negative 1 0 0 1 1
30398bf8373c_study,Negative 1 0 0 1 1
30ba38332bd6_study,Negative 1 0 0 1 1
31126c3051c9_study,Negative 1 0 0 1 1
313494ee0897_study,Negative 1 0 0 1 1
315b599c0470_study,Negative 1 0 0 1 1
3197b5b1d2bf_study,Negative 1 0 0 1 1
31f14fb6667d_study,Negative 1 0 0 1 1
32699739f719_study,Negative 1 0 0 1 1
326cdf61e601_study,Negative 1 0 0 1 1
32cd00648c62_study,Negative 1 0 0 1 1
32d24b5596d7_study,Negative 1 0 0 1 1
32d36978749c_study,Negative 1 0 0 1 1
334a4202e414_study,Negative 1 0 0 1 1
3379f980e291_study,Negative 1 0 0 1 1
337f24b92040_study,Negative 1 0 0 1 1
33cea11fd803_study,Negative 1 0 0 1 1
33cfd515b130_study,Negative 1 0 0 1 1
33df7cf3b609_study,Negative 1 0 0 1 1
33e1be3d2bae_study,Negative 1 0 0 1 1
342624fb2b11_study,Negative 1 0 0 1 1
343d755d7ddd_study,Negative 1 0 0 1 1
3456c012f2a8_study,Negative 1 0 0 1 1
3459b9abab30_study,Negative 1 0 0 1 1
34b78006dfce_study,Negative 1 0 0 1 1
34cfefd606be_study,Negative 1 0 0 1 1
34e297aea58b_study,Negative 1 0 0 1 1
34fc8c6bbfae_study,Negative 1 0 0 1 1
35036fa47dec_study,Negative 1 0 0 1 1
354d14a98d15_study,Negative 1 0 0 1 1
35a8a365b5cf_study,Negative 1 0 0 1 1
35c3971dfdf2_study,Negative 1 0 0 1 1
35c4922b2b72_study,Negative 1 0 0 1 1
364b50a7e35a_study,Negative 1 0 0 1 1
36623c43469f_study,Negative 1 0 0 1 1
3674dc34094e_study,Negative 1 0 0 1 1
36fabb793a4b_study,Negative 1 0 0 1 1
373876fd24c8_study,Negative 1 0 0 1 1
3745c672078c_study,Negative 1 0 0 1 1
37b217b51279_study,Negative 1 0 0 1 1
37b438e85030_study,Negative 1 0 0 1 1
37e448215647_study,Negative 1 0 0 1 1
37fc5147ab39_study,Negative 1 0 0 1 1
3848b7af690f_study,Negative 1 0 0 1 1
38dd9e92f0d0_study,Negative 1 0 0 1 1
394e0e1481e9_study,Negative 1 0 0 1 1
3997db0588ee_study,Negative 1 0 0 1 1
39a02fb99c60_study,Negative 1 0 0 1 1
3a02eb0ad5f0_study,Negative 1 0 0 1 1
3a04a4cbbe0f_study,Negative 1 0 0 1 1
3a1402e40308_study,Negative 1 0 0 1 1
3aac75b8f284_study,Negative 1 0 0 1 1
3af8cf394eaa_study,Negative 1 0 0 1 1
3afd951b907a_study,Negative 1 0 0 1 1
3b75f6371910_study,Negative 1 0 0 1 1
3bc2eec8d9b6_study,Negative 1 0 0 1 1
3bd22742cef2_study,Negative 1 0 0 1 1
3bdfb239eeaa_study,Negative 1 0 0 1 1
3be961dff18a_study,Negative 1 0 0 1 1
3c17fa476f85_study,Negative 1 0 0 1 1
3c214bbb0547_study,Negative 1 0 0 1 1
3c46f027da8c_study,Negative 1 0 0 1 1
3c4848aecb76_study,Negative 1 0 0 1 1
3c5cb14e548f_study,Negative 1 0 0 1 1
3c5de6b3939b_study,Negative 1 0 0 1 1
3c7b3b27ae98_study,Negative 1 0 0 1 1
3ca544da198b_study,Negative 1 0 0 1 1
3d3ad20f3435_study,Negative 1 0 0 1 1
3d9c9daa3178_study,Negative 1 0 0 1 1
3da0dc1f68ed_study,Negative 1 0 0 1 1
3e300ac96c98_study,Negative 1 0 0 1 1
3e5052f6edc9_study,Negative 1 0 0 1 1
3e5b5906bc1d_study,Negative 1 0 0 1 1
3e5f0fae3649_study,Negative 1 0 0 1 1
3e69bb4cfeb4_study,Negative 1 0 0 1 1
3f0b1b5976e8_study,Negative 1 0 0 1 1
3f3a09fedfba_study,Negative 1 0 0 1 1
3f8d8daf5eda_study,Negative 1 0 0 1 1
3fa81e73dcfd_study,Negative 1 0 0 1 1
3fb4aafae1d0_study,Negative 1 0 0 1 1
4019c1fe14bf_study,Negative 1 0 0 1 1
4048c25c7ef0_study,Negative 1 0 0 1 1
408679472ef6_study,Negative 1 0 0 1 1
40af0e78e972_study,Negative 1 0 0 1 1
411cfa759a2c_study,Negative 1 0 0 1 1
41429b19d479_study,Negative 1 0 0 1 1
416fb9b31385_study,Negative 1 0 0 1 1
418be6346093_study,Negative 1 0 0 1 1
420aecd56033_study,Negative 1 0 0 1 1
423ee88658a0_study,Negative 1 0 0 1 1
4252e9b23d04_study,Negative 1 0 0 1 1
4269acadeebc_study,Negative 1 0 0 1 1
42ad262ec9f1_study,Negative 1 0 0 1 1
42bab681437d_study,Negative 1 0 0 1 1
42d73bc891e1_study,Negative 1 0 0 1 1
431ebffa3b24_study,Negative 1 0 0 1 1
4349fdb4ea03_study,Negative 1 0 0 1 1
436a560afe62_study,Negative 1 0 0 1 1
439840f0770f_study,Negative 1 0 0 1 1
43d60e93cbf3_study,Negative 1 0 0 1 1
44115fd3174b_study,Negative 1 0 0 1 1
442bd0085586_study,Negative 1 0 0 1 1
444d139efc93_study,Negative 1 0 0 1 1
449805191715_study,Negative 1 0 0 1 1
4530b0ca0c8f_study,Negative 1 0 0 1 1
45a75e62f53e_study,Negative 1 0 0 1 1
45e65e76f35b_study,Negative 1 0 0 1 1
4645169e41f1_study,Negative 1 0 0 1 1
466f38a39d33_study,Negative 1 0 0 1 1
4692315d67f9_study,Negative 1 0 0 1 1
46f330e7df31_study,Negative 1 0 0 1 1
47a9f63bf953_study,Negative 1 0 0 1 1
47fcb7fc49b4_study,Negative 1 0 0 1 1
482f2d224527_study,Negative 1 0 0 1 1
4831aac65647_study,Negative 1 0 0 1 1
48607ea71306_study,Negative 1 0 0 1 1
4894244507de_study,Negative 1 0 0 1 1
48d36324c0fc_study,Negative 1 0 0 1 1
4916eadc694d_study,Negative 1 0 0 1 1
492159ddba17_study,Negative 1 0 0 1 1
49648f2a8c51_study,Negative 1 0 0 1 1
4974db1937a3_study,Negative 1 0 0 1 1
4996bf5117c7_study,Negative 1 0 0 1 1
49b4180be546_study,Negative 1 0 0 1 1
4a27dab6f704_study,Negative 1 0 0 1 1
4a4334902cd0_study,Negative 1 0 0 1 1
4a466be6275b_study,Negative 1 0 0 1 1
4acb2c6bae62_study,Negative 1 0 0 1 1
4ae05ce074ad_study,Negative 1 0 0 1 1
4b4e09317bc8_study,Negative 1 0 0 1 1
4c2813caa0fb_study,Negative 1 0 0 1 1
4c7af4bdfbb2_study,Negative 1 0 0 1 1
4c8a23708cf3_study,Negative 1 0 0 1 1
4cb59fb01420_study,Negative 1 0 0 1 1
4cbdd2d5ca8f_study,Negative 1 0 0 1 1
4cc67741b7be_study,Negative 1 0 0 1 1
4d10b356cb2b_study,Negative 1 0 0 1 1
4d1347d7d63e_study,Negative 1 0 0 1 1
4d5b70ca0c23_study,Negative 1 0 0 1 1
4dee79301687_study,Negative 1 0 0 1 1
4e4ee0341fab_study,Negative 1 0 0 1 1
4ecd8728c643_study,Negative 1 0 0 1 1
4ede1ea5de26_study,Negative 1 0 0 1 1
4ee42fdc7d39_study,Negative 1 0 0 1 1
4f2795c6c431_study,Negative 1 0 0 1 1
4f3bd784d1bf_study,Negative 1 0 0 1 1
4f693f6cf408_study,Negative 1 0 0 1 1
4f9c9296104b_study,Negative 1 0 0 1 1
4fa25a4ab48b_study,Negative 1 0 0 1 1
4fdb1658707b_study,Negative 1 0 0 1 1
500d963baa00_study,Negative 1 0 0 1 1
504cf4d936de_study,Negative 1 0 0 1 1
50501d03ea2a_study,Negative 1 0 0 1 1
505b1f7bbbf4_study,Negative 1 0 0 1 1
50a27d21b5c5_study,Negative 1 0 0 1 1
51160c75a305_study,Negative 1 0 0 1 1
513506120b3e_study,Negative 1 0 0 1 1
514c5133572d_study,Negative 1 0 0 1 1
51f1367f7d9c_study,Negative 1 0 0 1 1
52466d50101e_study,Negative 1 0 0 1 1
525e881d45da_study,Negative 1 0 0 1 1
52950e68ba64_study,Negative 1 0 0 1 1
529b4311adb6_study,Negative 1 0 0 1 1
52e603bf575f_study,Negative 1 0 0 1 1
531912276b32_study,Negative 1 0 0 1 1
531aa20ff7c3_study,Negative 1 0 0 1 1
53e7a0aa4022_study,Negative 1 0 0 1 1
5461386157ed_study,Negative 1 0 0 1 1
5462f84903c3_study,Negative 1 0 0 1 1
5473b7261b97_study,Negative 1 0 0 1 1
54ab1c9162d3_study,Negative 1 0 0 1 1
54cea7959efa_study,Negative 1 0 0 1 1
54dc0eb52a0a_study,Negative 1 0 0 1 1
5527c97636a9_study,Negative 1 0 0 1 1
5566d3f34fbb_study,Negative 1 0 0 1 1
556ac9754637_study,Negative 1 0 0 1 1
556ed339d8a7_study,Negative 1 0 0 1 1
55f2e8f96a24_study,Negative 1 0 0 1 1
55ffd8d1a60a_study,Negative 1 0 0 1 1
56149fff7475_study,Negative 1 0 0 1 1
562bda48ad02_study,Negative 1 0 0 1 1
566a2983842a_study,Negative 1 0 0 1 1
566ce9e55829_study,Negative 1 0 0 1 1
56b339277951_study,Negative 1 0 0 1 1
5702713aeac0_study,Negative 1 0 0 1 1
57118f0c7115_study,Negative 1 0 0 1 1
57195fc6fe41_study,Negative 1 0 0 1 1
572fa8640acc_study,Negative 1 0 0 1 1
5764dee0ac43_study,Negative 1 0 0 1 1
579a4dfc64a2_study,Negative 1 0 0 1 1
57b682c96f90_study,Negative 1 0 0 1 1
57dad1ccc875_study,Negative 1 0 0 1 1
57e2c6e368f2_study,Negative 1 0 0 1 1
582f1d954423_study,Negative 1 0 0 1 1
588f2c4a8983_study,Negative 1 0 0 1 1
58d5f7145d50_study,Negative 1 0 0 1 1
593c3f815635_study,Negative 1 0 0 1 1
5956ba4a4433_study,Negative 1 0 0 1 1
5965818fa0e7_study,Negative 1 0 0 1 1
59834dbc2b91_study,Negative 1 0 0 1 1
5990b940ec49_study,Negative 1 0 0 1 1
5a74a91d9877_study,Negative 1 0 0 1 1
5a7b49158f30_study,Negative 1 0 0 1 1
5a9b482fce1f_study,Negative 1 0 0 1 1
5acd25132041_study,Negative 1 0 0 1 1
5ae19539344f_study,Negative 1 0 0 1 1
5b42cb9ee77b_study,Negative 1 0 0 1 1
5b4a49693706_study,Negative 1 0 0 1 1
5be3004f3c87_study,Negative 1 0 0 1 1
5bf4446c7cea_study,Negative 1 0 0 1 1
5cb07dad2ba7_study,Negative 1 0 0 1 1
5cd8e9cd57f4_study,Negative 1 0 0 1 1
5cecf76081cc_study,Negative 1 0 0 1 1
5d2582cd935e_study,Negative 1 0 0 1 1
5d6a3bfece97_study,Negative 1 0 0 1 1
5d71a4f40898_study,Negative 1 0 0 1 1
5d9ff25b7960_study,Negative 1 0 0 1 1
5dc5caf41360_study,Negative 1 0 0 1 1
5de32bda4c7a_study,Negative 1 0 0 1 1
5defaf7b006e_study,Negative 1 0 0 1 1
5e656faf76ea_study,Negative 1 0 0 1 1
5e729ae9ae3c_study,Negative 1 0 0 1 1
5e99801f484c_study,Negative 1 0 0 1 1
5ed4ac4c2302_study,Negative 1 0 0 1 1
5f04f4c0fa68_study,Negative 1 0 0 1 1
5f0e6a9f6f6d_study,Negative 1 0 0 1 1
5f14606437b6_study,Negative 1 0 0 1 1
5f46958edec1_study,Negative 1 0 0 1 1
5f6bd96b250b_study,Negative 1 0 0 1 1
5f8b1e975bdf_study,Negative 1 0 0 1 1
5fec26943d57_study,Negative 1 0 0 1 1
60a72f493074_study,Negative 1 0 0 1 1
611415679b0b_study,Negative 1 0 0 1 1
6117058c3931_study,Negative 1 0 0 1 1
616f2fcba772_study,Negative 1 0 0 1 1
61bc50228c92_study,Negative 1 0 0 1 1
61cab278c091_study,Negative 1 0 0 1 1
622f5b348f80_study,Negative 1 0 0 1 1
62659a70b6d2_study,Negative 1 0 0 1 1
6267862574f2_study,Negative 1 0 0 1 1
62912425183c_study,Negative 1 0 0 1 1
62a423696785_study,Negative 1 0 0 1 1
635b0ef5b3b9_study,Negative 1 0 0 1 1
635c78729d10_study,Negative 1 0 0 1 1
6386d34fe45f_study,Negative 1 0 0 1 1
63946400d861_study,Negative 1 0 0 1 1
63ba2f6870f1_study,Negative 1 0 0 1 1
64047404bb75_study,Negative 1 0 0 1 1
64068f878d2b_study,Negative 1 0 0 1 1
640e92dc7aa9_study,Negative 1 0 0 1 1
644ae1050800_study,Negative 1 0 0 1 1
64729a1c868e_study,Negative 1 0 0 1 1
6475a5f7bd09_study,Negative 1 0 0 1 1
6484393291ea_study,Negative 1 0 0 1 1
6515b912dc6b_study,Negative 1 0 0 1 1
655eaa76e116_study,Negative 1 0 0 1 1
66300f02d92a_study,Negative 1 0 0 1 1
665aa3146373_study,Negative 1 0 0 1 1
6677129718eb_study,Negative 1 0 0 1 1
667eb7504965_study,Negative 1 0 0 1 1
66844043bb82_study,Negative 1 0 0 1 1
66d364215c35_study,Negative 1 0 0 1 1
677ad87244f5_study,Negative 1 0 0 1 1
67b808cc8cfc_study,Negative 1 0 0 1 1
67c5a09a1950_study,Negative 1 0 0 1 1
68d002822185_study,Negative 1 0 0 1 1
68d273e07961_study,Negative 1 0 0 1 1
690a42c7eded_study,Negative 1 0 0 1 1
694432509c27_study,Negative 1 0 0 1 1
6951842a2430_study,Negative 1 0 0 1 1
6958d88e7d72_study,Negative 1 0 0 1 1
6958fb143029_study,Negative 1 0 0 1 1
69fc686a4321_study,Negative 1 0 0 1 1
6a7ebacd5830_study,Negative 1 0 0 1 1
6ab8658d127c_study,Negative 1 0 0 1 1
6b28f1d59cb3_study,Negative 1 0 0 1 1
6b8a59647e54_study,Negative 1 0 0 1 1
6b9a744fb067_study,Negative 1 0 0 1 1
6ba2dee3517c_study,Negative 1 0 0 1 1
6bbe6a05947d_study,Negative 1 0 0 1 1
6bd8ee85ab15_study,Negative 1 0 0 1 1
6c4b79292e1b_study,Negative 1 0 0 1 1
6d9aa96e6350_study,Negative 1 0 0 1 1
6da6b1e0a269_study,Negative 1 0 0 1 1
6dc6d5660243_study,Negative 1 0 0 1 1
6ddd518ee32b_study,Negative 1 0 0 1 1
6deb9fe72007_study,Negative 1 0 0 1 1
6decab09779c_study,Negative 1 0 0 1 1
6e652d81f5bf_study,Negative 1 0 0 1 1
6eab8786fa4a_study,Negative 1 0 0 1 1
6ebb505c89f9_study,Negative 1 0 0 1 1
6ed0d449ff97_study,Negative 1 0 0 1 1
6ef0c3dad8c6_study,Negative 1 0 0 1 1
6f5f6f99fb89_study,Negative 1 0 0 1 1
6f6e6f503df9_study,Negative 1 0 0 1 1
6fa6c83e0221_study,Negative 1 0 0 1 1
6ffe46db6f5c_study,Negative 1 0 0 1 1
703903704231_study,Negative 1 0 0 1 1
705257ddc51b_study,Negative 1 0 0 1 1
70a586a7a276_study,Negative 1 0 0 1 1
70f5376d8dd1_study,Negative 1 0 0 1 1
70f8ea19b10b_study,Negative 1 0 0 1 1
711eadb00212_study,Negative 1 0 0 1 1
716c5738a858_study,Negative 1 0 0 1 1
719ffac176ee_study,Negative 1 0 0 1 1
71beaaa0422e_study,Negative 1 0 0 1 1
71dbb924fb5c_study,Negative 1 0 0 1 1
7208d457bcc9_study,Negative 1 0 0 1 1
723c9ea28d31_study,Negative 1 0 0 1 1
728187b3e018_study,Negative 1 0 0 1 1
72a889b778b2_study,Negative 1 0 0 1 1
72e571a4dbcf_study,Negative 1 0 0 1 1
73094e54f847_study,Negative 1 0 0 1 1
7325a59e75b3_study,Negative 1 0 0 1 1
735c7f97b72f_study,Negative 1 0 0 1 1
737adbb5ee4d_study,Negative 1 0 0 1 1
73e36d50744d_study,Negative 1 0 0 1 1
74030b79b6ff_study,Negative 1 0 0 1 1
74448343f85a_study,Negative 1 0 0 1 1
749e75b3c993_study,Negative 1 0 0 1 1
74a3c66551b6_study,Negative 1 0 0 1 1
7516cd3f2680_study,Negative 1 0 0 1 1
7518068f3699_study,Negative 1 0 0 1 1
755a8831463d_study,Negative 1 0 0 1 1
7568cb61374c_study,Negative 1 0 0 1 1
7591c93cf38c_study,Negative 1 0 0 1 1
75e3c04fed9f_study,Negative 1 0 0 1 1
760466ddce29_study,Negative 1 0 0 1 1
766c04246e24_study,Negative 1 0 0 1 1
771e394e7e37_study,Negative 1 0 0 1 1
778ca3c0e198_study,Negative 1 0 0 1 1
77bea1a5baf9_study,Negative 1 0 0 1 1
780517944267_study,Negative 1 0 0 1 1
781c93fa2104_study,Negative 1 0 0 1 1
78685ad6af53_study,Negative 1 0 0 1 1
78e8b6f84907_study,Negative 1 0 0 1 1
78ff750810bf_study,Negative 1 0 0 1 1
790e1425ea5e_study,Negative 1 0 0 1 1
793fc16d2eaa_study,Negative 1 0 0 1 1
795051254905_study,Negative 1 0 0 1 1
7981605927db_study,Negative 1 0 0 1 1
798e11b2c8c2_study,Negative 1 0 0 1 1
79ed25911104_study,Negative 1 0 0 1 1
7a08c2cdd552_study,Negative 1 0 0 1 1
7a2bbfb17503_study,Negative 1 0 0 1 1
7a2e48a54953_study,Negative 1 0 0 1 1
7a5665185f3c_study,Negative 1 0 0 1 1
7a7cd9fa7f09_study,Negative 1 0 0 1 1
7a7e66fc9b3c_study,Negative 1 0 0 1 1
7ac22c3830e8_study,Negative 1 0 0 1 1
7b2693269d7b_study,Negative 1 0 0 1 1
7b5c31f2af6f_study,Negative 1 0 0 1 1
7b99371b9ec4_study,Negative 1 0 0 1 1
7bb87ec91b7d_study,Negative 1 0 0 1 1
7c47d71d9141_study,Negative 1 0 0 1 1
7c8ea185b141_study,Negative 1 0 0 1 1
7cbd9aaac8ba_study,Negative 1 0 0 1 1
7ce8a0052810_study,Negative 1 0 0 1 1
7d2470b0005e_study,Negative 1 0 0 1 1
7d5c6509f13b_study,Negative 1 0 0 1 1
7d706788a91c_study,Negative 1 0 0 1 1
7da6e56c18b7_study,Negative 1 0 0 1 1
7dc1850a0f94_study,Negative 1 0 0 1 1
7e1d636a24f0_study,Negative 1 0 0 1 1
7e6845a52319_study,Negative 1 0 0 1 1
7e8981a09016_study,Negative 1 0 0 1 1
7ea66c84821e_study,Negative 1 0 0 1 1
7ea6876bfa96_study,Negative 1 0 0 1 1
7f188984feef_study,Negative 1 0 0 1 1
7f3115e66c1e_study,Negative 1 0 0 1 1
7f6449e6861c_study,Negative 1 0 0 1 1
7ff9928af7f6_study,Negative 1 0 0 1 1
8011f96848a6_study,Negative 1 0 0 1 1
803976b68795_study,Negative 1 0 0 1 1
8066c8d7cba7_study,Negative 1 0 0 1 1
806a948c30bf_study,Negative 1 0 0 1 1
809e196af923_study,Negative 1 0 0 1 1
80a77c5dd48a_study,Negative 1 0 0 1 1
80e1c35abd04_study,Negative 1 0 0 1 1
8128f5692b12_study,Negative 1 0 0 1 1
8137f72e23d2_study,Negative 1 0 0 1 1
817c43aedcce_study,Negative 1 0 0 1 1
81a119607e57_study,Negative 1 0 0 1 1
81c860c6efe8_study,Negative 1 0 0 1 1
81e6c9f2675c_study,Negative 1 0 0 1 1
820df4b6ce93_study,Negative 1 0 0 1 1
820f82abb614_study,Negative 1 0 0 1 1
82441f0b61c4_study,Negative 1 0 0 1 1
82453d7d1b90_study,Negative 1 0 0 1 1
82686606c173_study,Negative 1 0 0 1 1
827ce7be2a4e_study,Negative 1 0 0 1 1
82a4d7f6ea1e_study,Negative 1 0 0 1 1
82ec59c197d0_study,Negative 1 0 0 1 1
8309732e6809_study,Negative 1 0 0 1 1
830abea627fc_study,Negative 1 0 0 1 1
83195705dd21_study,Negative 1 0 0 1 1
8322b4b12bf2_study,Negative 1 0 0 1 1
8334d5ebdfd0_study,Negative 1 0 0 1 1
8353ed908a04_study,Negative 1 0 0 1 1
835ff4bc9c2b_study,Negative 1 0 0 1 1
83958be8c4b8_study,Negative 1 0 0 1 1
839c08f4b18a_study,Negative 1 0 0 1 1
83e34d429bd2_study,Negative 1 0 0 1 1
83e89d87256b_study,Negative 1 0 0 1 1
8452e137c3fb_study,Negative 1 0 0 1 1
847b7a65658a_study,Negative 1 0 0 1 1
84ade5a82135_study,Negative 1 0 0 1 1
84f431dbdbd6_study,Negative 1 0 0 1 1
852912afea0c_study,Negative 1 0 0 1 1
855945ea39f1_study,Negative 1 0 0 1 1
858b3a423bb3_study,Negative 1 0 0 1 1
85d692693bc1_study,Negative 1 0 0 1 1
85e02d7fac77_study,Negative 1 0 0 1 1
8613aa92c09d_study,Negative 1 0 0 1 1
865fedc5484d_study,Negative 1 0 0 1 1
86619ada1b6d_study,Negative 1 0 0 1 1
869b630384e5_study,Negative 1 0 0 1 1
8723a0e3b9e8_study,Negative 1 0 0 1 1
879a9c1601d7_study,Negative 1 0 0 1 1
87d6e4b3d6af_study,Negative 1 0 0 1 1
8822f2e6799c_study,Negative 1 0 0 1 1
8997cc349054_study,Negative 1 0 0 1 1
899e2e409330_study,Negative 1 0 0 1 1
89d2d0f18347_study,Negative 1 0 0 1 1
89ef92d3e02a_study,Negative 1 0 0 1 1
8a2e5381a43c_study,Negative 1 0 0 1 1
8a3b0b977e3d_study,Negative 1 0 0 1 1
8a6902767de9_study,Negative 1 0 0 1 1
8a6bce8f43ad_study,Negative 1 0 0 1 1
8a73dc1a0bfd_study,Negative 1 0 0 1 1
8a7c4530180a_study,Negative 1 0 0 1 1
8a8bc32ca39d_study,Negative 1 0 0 1 1
8ab03e394f4d_study,Negative 1 0 0 1 1
8ad5ead9c761_study,Negative 1 0 0 1 1
8b273337a684_study,Negative 1 0 0 1 1
8b61bfc4c50f_study,Negative 1 0 0 1 1
8b6e18ee2c40_study,Negative 1 0 0 1 1
8c1fd7170061_study,Negative 1 0 0 1 1
8c208e83154a_study,Negative 1 0 0 1 1
8c294e104651_study,Negative 1 0 0 1 1
8c6967991800_study,Negative 1 0 0 1 1
8ca5a21b8bd5_study,Negative 1 0 0 1 1
8ce2b4d54836_study,Negative 1 0 0 1 1
8d1c64890229_study,Negative 1 0 0 1 1
8d5e8712712d_study,Negative 1 0 0 1 1
8db23065b351_study,Negative 1 0 0 1 1
8deff6e2632d_study,Negative 1 0 0 1 1
8e07907d854d_study,Negative 1 0 0 1 1
8e1c060be66a_study,Negative 1 0 0 1 1
8e22a69566b2_study,Negative 1 0 0 1 1
8e639388fbce_study,Negative 1 0 0 1 1
8e6bc9d87d7b_study,Negative 1 0 0 1 1
8e90cf0a2220_study,Negative 1 0 0 1 1
8e99c311c641_study,Negative 1 0 0 1 1
8ead56410abf_study,Negative 1 0 0 1 1
8ec580935d86_study,Negative 1 0 0 1 1
8f15be256d40_study,Negative 1 0 0 1 1
8f19b8e12bb6_study,Negative 1 0 0 1 1
8f6df9fe9bae_study,Negative 1 0 0 1 1
8f6e72825177_study,Negative 1 0 0 1 1
8f8b690689ff_study,Negative 1 0 0 1 1
8fa44f4e8eeb_study,Negative 1 0 0 1 1
9004140ff428_study,Negative 1 0 0 1 1
905fdf23cba2_study,Negative 1 0 0 1 1
90ba8bbb45c4_study,Negative 1 0 0 1 1
91091b14ff14_study,Negative 1 0 0 1 1
9118eeb0ff4e_study,Negative 1 0 0 1 1
913836f6e106_study,Negative 1 0 0 1 1
91fd1001ca08_study,Negative 1 0 0 1 1
9260839a059d_study,Negative 1 0 0 1 1
92a0bb576754_study,Negative 1 0 0 1 1
92d320a0719d_study,Negative 1 0 0 1 1
92e806962d05_study,Negative 1 0 0 1 1
93a2b6809c3b_study,Negative 1 0 0 1 1
93b334bb7f73_study,Negative 1 0 0 1 1
94571b6e07ff_study,Negative 1 0 0 1 1
945c4c4bf174_study,Negative 1 0 0 1 1
94637b260217_study,Negative 1 0 0 1 1
94ab5762a56c_study,Negative 1 0 0 1 1
94b9dfb7e357_study,Negative 1 0 0 1 1
94fd8d61a3a2_study,Negative 1 0 0 1 1
952b24144e0c_study,Negative 1 0 0 1 1
956a8d5f7ab4_study,Negative 1 0 0 1 1
95d14f2a3b9c_study,Negative 1 0 0 1 1
95d39cd555f7_study,Negative 1 0 0 1 1
95fc78daff79_study,Negative 1 0 0 1 1
960176160f7f_study,Negative 1 0 0 1 1
96401434c121_study,Negative 1 0 0 1 1
9672b3805db6_study,Negative 1 0 0 1 1
968d50203714_study,Negative 1 0 0 1 1
96afb6a80094_study,Negative 1 0 0 1 1
96b3ffe98dd0_study,Negative 1 0 0 1 1
97c5d6eb413d_study,Negative 1 0 0 1 1
9805f2073fe2_study,Negative 1 0 0 1 1
983688dfdc6b_study,Negative 1 0 0 1 1
9852d2660b69_study,Negative 1 0 0 1 1
98603aec7e42_study,Negative 1 0 0 1 1
98962ec0ac81_study,Negative 1 0 0 1 1
99207112bd34_study,Negative 1 0 0 1 1
994aa93b88d6_study,Negative 1 0 0 1 1
9999ee344469_study,Negative 1 0 0 1 1
99abcf5c94b9_study,Negative 1 0 0 1 1
99b182a739fe_study,Negative 1 0 0 1 1
99f4549a0ce7_study,Negative 1 0 0 1 1
9a1a4fe68496_study,Negative 1 0 0 1 1
9a2421d1e810_study,Negative 1 0 0 1 1
9aae1d401336_study,Negative 1 0 0 1 1
9b02e142b13f_study,Negative 1 0 0 1 1
9b284d2b8e81_study,Negative 1 0 0 1 1
9b2f1505d1ba_study,Negative 1 0 0 1 1
9b74446eee7b_study,Negative 1 0 0 1 1
9b8eca950609_study,Negative 1 0 0 1 1
9bcfa30619cb_study,Negative 1 0 0 1 1
9beb97e23a51_study,Negative 1 0 0 1 1
9bfb252388d3_study,Negative 1 0 0 1 1
9c15ad2a3f00_study,Negative 1 0 0 1 1
9c7c9aa6b32e_study,Negative 1 0 0 1 1
9cd49495a7ca_study,Negative 1 0 0 1 1
9d88b7a41476_study,Negative 1 0 0 1 1
9d9a9218f7df_study,Negative 1 0 0 1 1
9e36f980af83_study,Negative 1 0 0 1 1
9e4fbb2efbf9_study,Negative 1 0 0 1 1
9e7cbbecfcda_study,Negative 1 0 0 1 1
9eb632c3beb5_study,Negative 1 0 0 1 1
9ec66102e783_study,Negative 1 0 0 1 1
9fab41ffbc39_study,Negative 1 0 0 1 1
9fd32ab29bfa_study,Negative 1 0 0 1 1
a03006dd606d_study,Negative 1 0 0 1 1
a09216d3b1fb_study,Negative 1 0 0 1 1
a0b1e8cfd597_study,Negative 1 0 0 1 1
a0b88a7bbda9_study,Negative 1 0 0 1 1
a134c7f3e533_study,Negative 1 0 0 1 1
a150ce575fd8_study,Negative 1 0 0 1 1
a15e73c6df9f_study,Negative 1 0 0 1 1
a167e3cfc945_study,Negative 1 0 0 1 1
a1aa5f84e8df_study,Negative 1 0 0 1 1
a1c32654a524_study,Negative 1 0 0 1 1
a1d7141d8672_study,Negative 1 0 0 1 1
a2335d0b5319_study,Negative 1 0 0 1 1
a24ebf2b7a7b_study,Negative 1 0 0 1 1
a28c3ca0b20d_study,Negative 1 0 0 1 1
a30f9f093480_study,Negative 1 0 0 1 1
a3277b063327_study,Negative 1 0 0 1 1
a32b55f395fb_study,Negative 1 0 0 1 1
a3489d5c2d26_study,Negative 1 0 0 1 1
a375466fac0b_study,Negative 1 0 0 1 1
a383dd18ad9b_study,Negative 1 0 0 1 1
a39a5337c940_study,Negative 1 0 0 1 1
a4798ae26b6e_study,Negative 1 0 0 1 1
a49fa9f919e0_study,Negative 1 0 0 1 1
a4d6e1d14076_study,Negative 1 0 0 1 1
a4dec0e99eb8_study,Negative 1 0 0 1 1
a4f8482aa131_study,Negative 1 0 0 1 1
a5979058f5f9_study,Negative 1 0 0 1 1
a5f2750581c1_study,Negative 1 0 0 1 1
a5f572f88efc_study,Negative 1 0 0 1 1
a60551308849_study,Negative 1 0 0 1 1
a655f779bc10_study,Negative 1 0 0 1 1
a6999828675a_study,Negative 1 0 0 1 1
a6b745712821_study,Negative 1 0 0 1 1
a6cd526c279a_study,Negative 1 0 0 1 1
a707e083e349_study,Negative 1 0 0 1 1
a73262c7ff37_study,Negative 1 0 0 1 1
a73f225845e3_study,Negative 1 0 0 1 1
a7526a74ab03_study,Negative 1 0 0 1 1
a7fdeed27109_study,Negative 1 0 0 1 1
a80006bd7d8f_study,Negative 1 0 0 1 1
a80b17b9947b_study,Negative 1 0 0 1 1
a82c2df28cbc_study,Negative 1 0 0 1 1
a853ec3e7735_study,Negative 1 0 0 1 1
a882dd5cb2de_study,Negative 1 0 0 1 1
a8c159935334_study,Negative 1 0 0 1 1
a8fe3043e449_study,Negative 1 0 0 1 1
a91ca6e3bc53_study,Negative 1 0 0 1 1
a9a0208dcbe3_study,Negative 1 0 0 1 1
a9b90530258c_study,Negative 1 0 0 1 1
a9c37909c52a_study,Negative 1 0 0 1 1
a9d35b0f34af_study,Negative 1 0 0 1 1
aa1013c021d7_study,Negative 1 0 0 1 1
aa1570677120_study,Negative 1 0 0 1 1
aac24f8be440_study,Negative 1 0 0 1 1
aafc8126d5a1_study,Negative 1 0 0 1 1
ab405aab1bf4_study,Negative 1 0 0 1 1
ab44c2e2e8ab_study,Negative 1 0 0 1 1
ab812073c00d_study,Negative 1 0 0 1 1
abbd693aab76_study,Negative 1 0 0 1 1
abf963944680_study,Negative 1 0 0 1 1
ac0854e5cbf6_study,Negative 1 0 0 1 1
ac5ce91c09e1_study,Negative 1 0 0 1 1
ac96e9b9034d_study,Negative 1 0 0 1 1
acc0ac018b57_study,Negative 1 0 0 1 1
acc1e6bcca9a_study,Negative 1 0 0 1 1
acc21b1fad3a_study,Negative 1 0 0 1 1
accb45de0444_study,Negative 1 0 0 1 1
ad0172d99745_study,Negative 1 0 0 1 1
ad4d579a3eab_study,Negative 1 0 0 1 1
ad61aee2f837_study,Negative 1 0 0 1 1
ad6731c3a452_study,Negative 1 0 0 1 1
ade239c090a7_study,Negative 1 0 0 1 1
ae0ba9f54467_study,Negative 1 0 0 1 1
ae2753d01a5d_study,Negative 1 0 0 1 1
ae40389b1d61_study,Negative 1 0 0 1 1
ae8fadd32d3e_study,Negative 1 0 0 1 1
aeb371eab69a_study,Negative 1 0 0 1 1
aeb50bd2a339_study,Negative 1 0 0 1 1
af09deb4a6f7_study,Negative 1 0 0 1 1
af12c30f7051_study,Negative 1 0 0 1 1
af3baa41780a_study,Negative 1 0 0 1 1
af619aba9623_study,Negative 1 0 0 1 1
af8cc211df75_study,Negative 1 0 0 1 1
b05b84f90ccf_study,Negative 1 0 0 1 1
b07759d0c81c_study,Negative 1 0 0 1 1
b0881924653b_study,Negative 1 0 0 1 1
b0e3eb0ced34_study,Negative 1 0 0 1 1
b11776be9437_study,Negative 1 0 0 1 1
b12171726b36_study,Negative 1 0 0 1 1
b12d2e5c39b7_study,Negative 1 0 0 1 1
b16a2caa5a14_study,Negative 1 0 0 1 1
b1b1ac90f30e_study,Negative 1 0 0 1 1
b273193222c5_study,Negative 1 0 0 1 1
b288b9700df7_study,Negative 1 0 0 1 1
b2ae9049c886_study,Negative 1 0 0 1 1
b2b95741c954_study,Negative 1 0 0 1 1
b2ef21c21a69_study,Negative 1 0 0 1 1
b378e3902756_study,Negative 1 0 0 1 1
b382d866a3f4_study,Negative 1 0 0 1 1
b4091ecc7f5d_study,Negative 1 0 0 1 1
b43b067ad5fc_study,Negative 1 0 0 1 1
b4436e4a91bc_study,Negative 1 0 0 1 1
b4552ba03116_study,Negative 1 0 0 1 1
b45b2ff7b925_study,Negative 1 0 0 1 1
b4ed20f4c603_study,Negative 1 0 0 1 1
b5052bc9081c_study,Negative 1 0 0 1 1
b53b95b262c5_study,Negative 1 0 0 1 1
b53ff1b3e45d_study,Negative 1 0 0 1 1
b5649ea5d97e_study,Negative 1 0 0 1 1
b59b6ffdecdc_study,Negative 1 0 0 1 1
b5d64243c33e_study,Negative 1 0 0 1 1
b5dc37ecfd1a_study,Negative 1 0 0 1 1
b60089258936_study,Negative 1 0 0 1 1
b61a3dba7e14_study,Negative 1 0 0 1 1
b641f8bc6e83_study,Negative 1 0 0 1 1
b647d5c8422e_study,Negative 1 0 0 1 1
b64b5a8cec28_study,Negative 1 0 0 1 1
b66d6c34498e_study,Negative 1 0 0 1 1
b6deca486769_study,Negative 1 0 0 1 1
b6ef87588b85_study,Negative 1 0 0 1 1
b70e9d1f6548_study,Negative 1 0 0 1 1
b7d09941c6c7_study,Negative 1 0 0 1 1
b7e73e54a566_study,Negative 1 0 0 1 1
b818f246a8ab_study,Negative 1 0 0 1 1
b83eaac8a377_study,Negative 1 0 0 1 1
b8a3899a2ddb_study,Negative 1 0 0 1 1
b8abb3588261_study,Negative 1 0 0 1 1
b8b849c98dfa_study,Negative 1 0 0 1 1
b93bfa119338_study,Negative 1 0 0 1 1
b93db355ba6a_study,Negative 1 0 0 1 1
b948441f0c12_study,Negative 1 0 0 1 1
b994bed5dd12_study,Negative 1 0 0 1 1
b99e07512d7a_study,Negative 1 0 0 1 1
b9aaac6a5ee8_study,Negative 1 0 0 1 1
b9e6a216ca98_study,Negative 1 0 0 1 1
b9e8350b5fb2_study,Negative 1 0 0 1 1
b9e87bbcdf15_study,Negative 1 0 0 1 1
b9fdedd4051e_study,Negative 1 0 0 1 1
ba21fc21ec33_study,Negative 1 0 0 1 1
ba2c87bf8de9_study,Negative 1 0 0 1 1
ba477e03efc4_study,Negative 1 0 0 1 1
ba54b6a4d83d_study,Negative 1 0 0 1 1
ba66195d3d10_study,Negative 1 0 0 1 1
bac4b1b19508_study,Negative 1 0 0 1 1
baff0a97d334_study,Negative 1 0 0 1 1
bb33957a9bd5_study,Negative 1 0 0 1 1
bbbcb26b74be_study,Negative 1 0 0 1 1
bc22d07c4f95_study,Negative 1 0 0 1 1
bc2e72409e5f_study,Negative 1 0 0 1 1
bc73883551cf_study,Negative 1 0 0 1 1
bc76c0035143_study,Negative 1 0 0 1 1
bc7a40fef51c_study,Negative 1 0 0 1 1
bc8a085104af_study,Negative 1 0 0 1 1
bc95acc5c308_study,Negative 1 0 0 1 1
bc965b8e19f8_study,Negative 1 0 0 1 1
bd2f3ba76d62_study,Negative 1 0 0 1 1
bd70b1ab4fe0_study,Negative 1 0 0 1 1
bd84b0b9598e_study,Negative 1 0 0 1 1
be3bdfa28903_study,Negative 1 0 0 1 1
be56a042b9b1_study,Negative 1 0 0 1 1
be6f27a4528b_study,Negative 1 0 0 1 1
be926eae5b48_study,Negative 1 0 0 1 1
bea266e15217_study,Negative 1 0 0 1 1
bea575481ef3_study,Negative 1 0 0 1 1
becdc25d08c4_study,Negative 1 0 0 1 1
becf878baece_study,Negative 1 0 0 1 1
bf2de109e4be_study,Negative 1 0 0 1 1
bf41375c6d47_study,Negative 1 0 0 1 1
bfe0b7efdb2a_study,Negative 1 0 0 1 1
c01a83bc7f7d_study,Negative 1 0 0 1 1
c0924d22f592_study,Negative 1 0 0 1 1
c09fa807d4d0_study,Negative 1 0 0 1 1
c1161e21e98f_study,Negative 1 0 0 1 1
c1d3c478d4c9_study,Negative 1 0 0 1 1
c2472b630e54_study,Negative 1 0 0 1 1
c268ee0b80e6_study,Negative 1 0 0 1 1
c2f7a0a9e328_study,Negative 1 0 0 1 1
c34508d810cb_study,Negative 1 0 0 1 1
c38f6e4a8b57_study,Negative 1 0 0 1 1
c39536795ca2_study,Negative 1 0 0 1 1
c3cbb6e1ff0b_study,Negative 1 0 0 1 1
c3d5485c6637_study,Negative 1 0 0 1 1
c4359b917227_study,Negative 1 0 0 1 1
c456c89aeeca_study,Negative 1 0 0 1 1
c56b60a6c549_study,Negative 1 0 0 1 1
c5ea13200bd6_study,Negative 1 0 0 1 1
c66cd65fd2ec_study,Negative 1 0 0 1 1
c6ce9fc03d44_study,Negative 1 0 0 1 1
c714e99b4e36_study,Negative 1 0 0 1 1
c725cff962b5_study,Negative 1 0 0 1 1
c7270e362816_study,Negative 1 0 0 1 1
c731d74015f8_study,Negative 1 0 0 1 1
c7389760417a_study,Negative 1 0 0 1 1
c7787a3cfb14_study,Negative 1 0 0 1 1
c783118f9673_study,Negative 1 0 0 1 1
c798ba1c69e1_study,Negative 1 0 0 1 1
c7c4926465d5_study,Negative 1 0 0 1 1
c803c01dfa08_study,Negative 1 0 0 1 1
c82731b476d7_study,Negative 1 0 0 1 1
c849770966c1_study,Negative 1 0 0 1 1
c85581436d7c_study,Negative 1 0 0 1 1
c87c82925501_study,Negative 1 0 0 1 1
c8b7a0ae731e_study,Negative 1 0 0 1 1
c8bd35fb64e1_study,Negative 1 0 0 1 1
c9163fa32483_study,Negative 1 0 0 1 1
c987fd114725_study,Negative 1 0 0 1 1
ca20e0f1b5c9_study,Negative 1 0 0 1 1
ca221d324958_study,Negative 1 0 0 1 1
ca87ac7f7497_study,Negative 1 0 0 1 1
caeeede7f74d_study,Negative 1 0 0 1 1
cafa1dca9af7_study,Negative 1 0 0 1 1
cbc5e2a2efac_study,Negative 1 0 0 1 1
cc874d5c6431_study,Negative 1 0 0 1 1
ccbd6814114b_study,Negative 1 0 0 1 1
ccc80d9fd30f_study,Negative 1 0 0 1 1
cd0664cef5a7_study,Negative 1 0 0 1 1
cd1f0a0bd109_study,Negative 1 0 0 1 1
cd98127bb203_study,Negative 1 0 0 1 1
cddf3a089653_study,Negative 1 0 0 1 1
ce1f49e49440_study,Negative 1 0 0 1 1
ce2d48a26e5e_study,Negative 1 0 0 1 1
ce3e55217552_study,Negative 1 0 0 1 1
ced8fdaa9dba_study,Negative 1 0 0 1 1
ceed10e91962_study,Negative 1 0 0 1 1
cf57b6b204f8_study,Negative 1 0 0 1 1
cf5f6f4b8cf5_study,Negative 1 0 0 1 1
cfad493ac469_study,Negative 1 0 0 1 1
d0059448cf97_study,Negative 1 0 0 1 1
d025a0d95412_study,Negative 1 0 0 1 1
d0914f314664_study,Negative 1 0 0 1 1
d0934bcb74f1_study,Negative 1 0 0 1 1
d0a22a0954f7_study,Negative 1 0 0 1 1
d0b6342fc4c0_study,Negative 1 0 0 1 1
d11b750bff94_study,Negative 1 0 0 1 1
d11db59069df_study,Negative 1 0 0 1 1
d128283294c9_study,Negative 1 0 0 1 1
d16959ef2164_study,Negative 1 0 0 1 1
d1a15a468593_study,Negative 1 0 0 1 1
d1cbcb51e5ae_study,Negative 1 0 0 1 1
d2f1eb1b0bc5_study,Negative 1 0 0 1 1
d33c77a3df19_study,Negative 1 0 0 1 1
d3818fbf4c36_study,Negative 1 0 0 1 1
d3b7d7dc583b_study,Negative 1 0 0 1 1
d3b82893b94e_study,Negative 1 0 0 1 1
d3be4764ade5_study,Negative 1 0 0 1 1
d3c53c4222e2_study,Negative 1 0 0 1 1
d414d9603210_study,Negative 1 0 0 1 1
d427c22b488d_study,Negative 1 0 0 1 1
d460dc993ca7_study,Negative 1 0 0 1 1
d477d68f839f_study,Negative 1 0 0 1 1
d478ec4faf9d_study,Negative 1 0 0 1 1
d47b71c4fcea_study,Negative 1 0 0 1 1
d48708c36bd5_study,Negative 1 0 0 1 1
d4c399afd466_study,Negative 1 0 0 1 1
d50e06374ee8_study,Negative 1 0 0 1 1
d50e37efdecd_study,Negative 1 0 0 1 1
d542fb6a6c0a_study,Negative 1 0 0 1 1
d566c844260f_study,Negative 1 0 0 1 1
d6104076541a_study,Negative 1 0 0 1 1
d6195f41dfbc_study,Negative 1 0 0 1 1
d63229f78c7e_study,Negative 1 0 0 1 1
d675f6d53f58_study,Negative 1 0 0 1 1
d692c60e21ef_study,Negative 1 0 0 1 1
d6ca5d0cc911_study,Negative 1 0 0 1 1
d7f4e3e0681d_study,Negative 1 0 0 1 1
d81ff6b98fea_study,Negative 1 0 0 1 1
d84b500fb378_study,Negative 1 0 0 1 1
d859c2fa0d50_study,Negative 1 0 0 1 1
d85e078e410a_study,Negative 1 0 0 1 1
d89e30d51e89_study,Negative 1 0 0 1 1
d8cf28d0aa53_study,Negative 1 0 0 1 1
d945e8dd9670_study,Negative 1 0 0 1 1
d99a3d8578f8_study,Negative 1 0 0 1 1
d9fd3c3feb5e_study,Negative 1 0 0 1 1
da2f118facc6_study,Negative 1 0 0 1 1
da352dfd1456_study,Negative 1 0 0 1 1
da56f1cee61b_study,Negative 1 0 0 1 1
da88fa8e989e_study,Negative 1 0 0 1 1
dabd4c35f18f_study,Negative 1 0 0 1 1
dad1fab5736c_study,Negative 1 0 0 1 1
db4af70e370f_study,Negative 1 0 0 1 1
db6241a6c470_study,Negative 1 0 0 1 1
db629c44b07f_study,Negative 1 0 0 1 1
dbc236093102_study,Negative 1 0 0 1 1
dc0c112cd5c3_study,Negative 1 0 0 1 1
dc307d4d5204_study,Negative 1 0 0 1 1
dc953a56204f_study,Negative 1 0 0 1 1
dccdfc539a8e_study,Negative 1 0 0 1 1
dcf6503029e2_study,Negative 1 0 0 1 1
dd097443ed15_study,Negative 1 0 0 1 1
dd33c10f402b_study,Negative 1 0 0 1 1
dd4b97e0a4ed_study,Negative 1 0 0 1 1
dd6d8fdb8da0_study,Negative 1 0 0 1 1
dde3f480600a_study,Negative 1 0 0 1 1
ddf53d706c8e_study,Negative 1 0 0 1 1
de4d8d3b1944_study,Negative 1 0 0 1 1
dea84a337a80_study,Negative 1 0 0 1 1
decfe4c0afa6_study,Negative 1 0 0 1 1
def217a20092_study,Negative 1 0 0 1 1
df4910ccf64b_study,Negative 1 0 0 1 1
df9a0954caef_study,Negative 1 0 0 1 1
dfa767f1395d_study,Negative 1 0 0 1 1
e058e2a40a5f_study,Negative 1 0 0 1 1
e0a30517364f_study,Negative 1 0 0 1 1
e0b57c12175d_study,Negative 1 0 0 1 1
e12356c87019_study,Negative 1 0 0 1 1
e125f3618cb4_study,Negative 1 0 0 1 1
e13c1f49c1fc_study,Negative 1 0 0 1 1
e15b18e5370f_study,Negative 1 0 0 1 1
e1e07acfa4f8_study,Negative 1 0 0 1 1
e1fd75be3587_study,Negative 1 0 0 1 1
e311763786b9_study,Negative 1 0 0 1 1
e3129503fbc1_study,Negative 1 0 0 1 1
e326c6e0c67d_study,Negative 1 0 0 1 1
e3b205144584_study,Negative 1 0 0 1 1
e3c0daf79960_study,Negative 1 0 0 1 1
e42e21814d68_study,Negative 1 0 0 1 1
e43ccd956545_study,Negative 1 0 0 1 1
e43d15f662f9_study,Negative 1 0 0 1 1
e457c0603507_study,Negative 1 0 0 1 1
e45ae11bca4e_study,Negative 1 0 0 1 1
e53d2d6ab600_study,Negative 1 0 0 1 1
e5726ad872ec_study,Negative 1 0 0 1 1
e5757dbb5303_study,Negative 1 0 0 1 1
e5eb6f1abf7f_study,Negative 1 0 0 1 1
e612de80f0f9_study,Negative 1 0 0 1 1
e61fe43cdbd8_study,Negative 1 0 0 1 1
e655a9293c63_study,Negative 1 0 0 1 1
e66a31f2dce3_study,Negative 1 0 0 1 1
e6b59e765c23_study,Negative 1 0 0 1 1
e6e02ec8aff5_study,Negative 1 0 0 1 1
e6ed3bdd8a48_study,Negative 1 0 0 1 1
e7070a04cc4c_study,Negative 1 0 0 1 1
e799f53d8348_study,Negative 1 0 0 1 1
e7c1a6d76d48_study,Negative 1 0 0 1 1
e89c29993812_study,Negative 1 0 0 1 1
e8bdafa1240a_study,Negative 1 0 0 1 1
e8d137c213a0_study,Negative 1 0 0 1 1
e9e993a2786a_study,Negative 1 0 0 1 1
ea2a62fbd279_study,Negative 1 0 0 1 1
ea33449554a5_study,Negative 1 0 0 1 1
ea8ed0380261_study,Negative 1 0 0 1 1
eaa6cf8327b0_study,Negative 1 0 0 1 1
eb70ae36e030_study,Negative 1 0 0 1 1
ec584e5c219e_study,Negative 1 0 0 1 1
ed257e125f16_study,Negative 1 0 0 1 1
ed55cb404718_study,Negative 1 0 0 1 1
ede8b619a446_study,Negative 1 0 0 1 1
edfe1e3b1fb8_study,Negative 1 0 0 1 1
ee7780677a6a_study,Negative 1 0 0 1 1
ee860264dd8c_study,Negative 1 0 0 1 1
ee9f48a701a6_study,Negative 1 0 0 1 1
eea1e56576f5_study,Negative 1 0 0 1 1
eea90eaf5391_study,Negative 1 0 0 1 1
eeb3e6ba2c29_study,Negative 1 0 0 1 1
ef3c5684ecae_study,Negative 1 0 0 1 1
ef3e7a0c9ce4_study,Negative 1 0 0 1 1
ef4b4ec56178_study,Negative 1 0 0 1 1
ef7cd0715fbb_study,Negative 1 0 0 1 1
ef93153d87b2_study,Negative 1 0 0 1 1
efc58b7cfb19_study,Negative 1 0 0 1 1
eff3315137e1_study,Negative 1 0 0 1 1
eff9e575ad15_study,Negative 1 0 0 1 1
f016d6bdebc1_study,Negative 1 0 0 1 1
f0883d979bdd_study,Negative 1 0 0 1 1
f0a380548c76_study,Negative 1 0 0 1 1
f0a3f409468a_study,Negative 1 0 0 1 1
f146eb3c25c2_study,Negative 1 0 0 1 1
f1be3cad11ef_study,Negative 1 0 0 1 1
f221f9f21445_study,Negative 1 0 0 1 1
f25cc47c3a0f_study,Negative 1 0 0 1 1
f29dcebea115_study,Negative 1 0 0 1 1
f2d6f31dc5a6_study,Negative 1 0 0 1 1
f3836378efa0_study,Negative 1 0 0 1 1
f39b840d4e6f_study,Negative 1 0 0 1 1
f3d5844aab79_study,Negative 1 0 0 1 1
f3e357df078b_study,Negative 1 0 0 1 1
f3f82683fcbc_study,Negative 1 0 0 1 1
f40497be74df_study,Negative 1 0 0 1 1
f41885c32732_study,Negative 1 0 0 1 1
f4dd822aa759_study,Negative 1 0 0 1 1
f5b981349e98_study,Negative 1 0 0 1 1
f5d4e4e1c82f_study,Negative 1 0 0 1 1
f601f8c9018c_study,Negative 1 0 0 1 1
f60630ffc310_study,Negative 1 0 0 1 1
f66894c11824_study,Negative 1 0 0 1 1
f6b80b77d753_study,Negative 1 0 0 1 1
f6bccef80118_study,Negative 1 0 0 1 1
f6de79c7f029_study,Negative 1 0 0 1 1
f71b52604902_study,Negative 1 0 0 1 1
f76a41789560_study,Negative 1 0 0 1 1
f78027b5a34e_study,Negative 1 0 0 1 1
f78df9f25143_study,Negative 1 0 0 1 1
f791fa30274e_study,Negative 1 0 0 1 1
f7bc207ff087_study,Negative 1 0 0 1 1
f7d3c8acc574_study,Negative 1 0 0 1 1
f7ddfea36bf0_study,Negative 1 0 0 1 1
f800558e2d67_study,Negative 1 0 0 1 1
f85b5d51e41d_study,Negative 1 0 0 1 1
f884821859a6_study,Negative 1 0 0 1 1
f8dbe97a5542_study,Negative 1 0 0 1 1
f928ae474f7e_study,Negative 1 0 0 1 1
f9e2dae3b0d1_study,Negative 1 0 0 1 1
faabf68ffc99_study,Negative 1 0 0 1 1
faaf340d46dc_study,Negative 1 0 0 1 1
faafe8fa3030_study,Negative 1 0 0 1 1
fae0a3a751a8_study,Negative 1 0 0 1 1
fb61db93b8da_study,Negative 1 0 0 1 1
fb6a6174b4cc_study,Negative 1 0 0 1 1
fb97c8dfe8e4_study,Negative 1 0 0 1 1
fbef288cd370_study,Negative 1 0 0 1 1
fbfbc8c07230_study,Negative 1 0 0 1 1
fc6054ce5985_study,Negative 1 0 0 1 1
fd15cbd7cced_study,Negative 1 0 0 1 1
fd193f9220cb_study,Negative 1 0 0 1 1
fd28c9e4074c_study,Negative 1 0 0 1 1
fd551460e2e4_study,Negative 1 0 0 1 1
fd90119fe61b_study,Negative 1 0 0 1 1
fd998ce80eb6_study,Negative 1 0 0 1 1
fdad33449860_study,Negative 1 0 0 1 1
fe3fa13c059c_study,Negative 1 0 0 1 1
fe48f2bc28b4_study,Negative 1 0 0 1 1
fe64182ae21d_study,Negative 1 0 0 1 1
fef37647517b_study,Negative 1 0 0 1 1
ff185ce05a5d_study,Negative 1 0 0 1 1
ff1ba0e9aaf0_study,Negative 1 0 0 1 1
ff2cc4de58c5_study,Negative 1 0 0 1 1
ff2f0a744930_study,Negative 1 0 0 1 1
ff88940dce8b_study,Negative 1 0 0 1 1
fff7ef24961f_study,Negative 1 0 0 1 1
557a70442928_image,None 1 0 0 1 1
36141cda67ad_image,None 1 0 0 1 1
2413a23a5477_image,opacity 0.472162 40 100 65 177 opacity 0.520418 147 221 78 194
c263b1e9aa64_image,opacity 0.33621 41 105 102 176 opacity 0.458538 170 228 72 176
4fe0444d7fc5_image,opacity 0.336574 179 243 106 166
1ff307ae0df6_image,None 1 0 0 1 1
b5e287db5781_image,None 1 0 0 1 1
07c552c36da3_image,None 1 0 0 1 1
c26e3ae08056_image,None 1 0 0 1 1
4eb9a8ab1559_image,None 1 0 0 1 1
afd960320c38_image,opacity 0.404012 64 128 87 203 opacity 0.493179 170 250 68 196
66d812201e73_image,None 1 0 0 1 1
864372de3df3_image,None 1 0 0 1 1
95381cfcbf0c_image,None 1 0 0 1 1
9cbc2fd041a7_image,opacity 0.344965 67 123 102 188
4d105a99d866_image,None 1 0 0 1 1
1863b31637f6_image,opacity 0.373036 136 192 67 175
e88f293bc83a_image,None 1 0 0 1 1
55ea6ce07bdd_image,opacity 0.403256 36 96 53 181 opacity 0.45379 146 204 77 169
cdc508b061fa_image,None 1 0 0 1 1
81646eb45af3_image,None 1 0 0 1 1
afce260dc959_image,opacity 0.522169 167 239 53 177 opacity 0.611821 52 124 49 173
e71a18669122_image,None 1 0 0 1 1
5dfad16f8d65_image,opacity 0.488907 153 209 42 176 opacity 0.524545 33 117 24 176
0cea7e93db89_image,opacity 0.489841 58 110 51 161
68b95eed58ed_image,None 1 0 0 1 1
5058f3b157e3_image,None 1 0 0 1 1
f1b95b972b74_image,None 1 0 0 1 1
f504d269be24_image,None 1 0 0 1 1
e00b7784d1eb_image,opacity 0.288509 168 222 128 182
ea7909e782fe_image,None 1 0 0 1 1
c836922a723b_image,None 1 0 0 1 1
aaef254ce2b3_image,None 1 0 0 1 1
9bee9808a9d4_image,opacity 0.495962 52 122 80 176
abedbc04a891_image,None 1 0 0 1 1
b08f26e45762_image,opacity 0.538193 47 107 63 167 opacity 0.619382 147 219 72 192
7e2e014e1656_image,None 1 0 0 1 1
9188ac2c54fe_image,None 1 0 0 1 1
c02ef6a70c7e_image,opacity 0.293307 59 111 73 177 opacity 0.318284 151 197 88 184
825dd8f7aca2_image,opacity 0.307999 31 83 74 170
65a02e95292e_image,None 1 0 0 1 1
9f1c1a909d7f_image,None 1 0 0 1 1
445298de90b9_image,opacity 0.523186 55 127 49 213 opacity 0.65954 161 241 40 192
2b7e9cf46376_image,opacity 0.456614 29 105 68 196 opacity 0.551855 136 208 53 177
08bbffaafbe7_image,opacity 0.285446 52 108 59 163
41251dd79bb3_image,None 1 0 0 1 1
1359178e428a_image,opacity 0.285913 61 117 105 169
1a1722362e19_image,None 1 0 0 1 1
2d58009c5ade_image,opacity 0.344706 24 80 71 167
7f4b304253ac_image,opacity 0.46686 153 217 72 176 opacity 0.522842 23 91 63 179
d3270449d5c5_image,None 1 0 0 1 1
4eecfe2a4de1_image,None 1 0 0 1 1
36b323299f7e_image,opacity 0.305979 153 207 126 226
78ff4d296adf_image,None 1 0 0 1 1
4041be34745f_image,opacity 0.312874 128 176 136 192
e78f51cdf0b8_image,opacity 0.411631 55 105 39 145
ea5341f767eb_image,None 1 0 0 1 1
aeaa44932cdd_image,None 1 0 0 1 1
ea79274e2bfd_image,None 1 0 0 1 1
6610e4632963_image,None 1 0 0 1 1
ba91d37ee459_image,opacity 0.446709 45 101 28 124 opacity 0.475661 151 205 36 148
e3a4dddc69b6_image,None 1 0 0 1 1
ab97e736a971_image,opacity 0.364906 20 84 81 249
b56f4eda9cea_image,None 1 0 0 1 1
317df13caea9_image,opacity 0.511709 146 214 50 146 opacity 0.628346 40 110 56 200
c24b6850df7d_image,None 1 0 0 1 1
aa6c8c46a05f_image,opacity 0.46578 153 225 117 233 opacity 0.48256 48 118 104 200
fa29fd163597_image,opacity 0.315049 79 133 105 197
d7cb763ec2b1_image,None 1 0 0 1 1
fdbaf93d6c5b_image,None 1 0 0 1 1
f17999420b3f_image,None 1 0 0 1 1
99b475d1d2ec_image,opacity 0.300386 164 252 50 212 opacity 0.412597 39 123 61 205
c28570a073d9_image,opacity 0.62455 170 226 54 194 opacity 0.659788 63 139 38 178
d1912847d707_image,opacity 0.371896 12 74 36 180
59d238641bee_image,None 1 0 0 1 1
ab46f325ad85_image,None 1 0 0 1 1
c3deaf98bf56_image,opacity 0.349257 152 208 116 228 opacity 0.409062 36 98 142 210
97324905f48f_image,opacity 0.355815 166 232 151 233
d438ffd35a00_image,opacity 0.460922 10 86 6 138 opacity 0.532911 137 197 47 163
9e63fbef3c21_image,None 1 0 0 1 1
eea4c86f42d5_image,None 1 0 0 1 1
966b86658ed3_image,opacity 0.281323 32 72 65 197
240bf89e4d58_image,opacity 0.289233 52 118 173 245 opacity 0.294698 172 244 113 249
6aedc81fb62d_image,None 1 0 0 1 1
11d59a5488ed_image,None 1 0 0 1 1
d7e440b4dc62_image,None 1 0 0 1 1
91faf22d3721_image,None 1 0 0 1 1
d832414be226_image,None 1 0 0 1 1
87a0829f53c1_image,opacity 0.387892 48 94 90 206
2910a7781437_image,None 1 0 0 1 1
179291642620_image,opacity 0.40395 49 113 41 191 opacity 0.623756 156 220 36 194
1e7429ce2e06_image,opacity 0.385141 65 111 46 172 opacity 0.429319 171 229 58 164
06df1e2fde68_image,opacity 0.420971 144 202 55 173 opacity 0.536237 48 104 54 170
efd8a04416af_image,None 1 0 0 1 1
b411d3c30168_image,None 1 0 0 1 1
88782677cbec_image,opacity 0.423566 70 130 132 234 opacity 0.503877 176 256 120 224
e32589cd4421_image,opacity 0.294971 30 98 30 136 opacity 0.491303 138 198 59 179
ea39ebfacca2_image,None 1 0 0 1 1
e5c56ef2d194_image,opacity 0.473656 31 87 64 170 opacity 0.545087 159 215 54 170
2dbd5f171a0f_image,None 1 0 0 1 1
e427d54f3926_image,None 1 0 0 1 1
ab471797d0a8_image,None 1 0 0 1 1
a9330037dce2_image,opacity 0.319596 156 204 138 206
7bf91ffc125c_image,None 1 0 0 1 1
19b0d2d65a38_image,None 1 0 0 1 1
51a0100c5921_image,None 1 0 0 1 1
7a09311ec3fa_image,None 1 0 0 1 1
9ad80e79362f_image,None 1 0 0 1 1
81bae2b0c027_image,opacity 0.416366 30 72 52 184
a9035a2d34f2_image,None 1 0 0 1 1
b0e7fab418bc_image,opacity 0.575083 62 122 35 167 opacity 0.582226 158 212 43 173
9b0ab5582681_image,None 1 0 0 1 1
926aeec260b1_image,opacity 0.444206 169 221 51 189
586b00a19a17_image,None 1 0 0 1 1
bbbd7ab57e75_image,None 1 0 0 1 1
0418a303883e_image,None 1 0 0 1 1
885136bc24c1_image,None 1 0 0 1 1
f038e6d16303_image,None 1 0 0 1 1
155f5e326d5e_image,opacity 0.374704 163 211 69 205 opacity 0.459545 62 114 121 201
305870c916a4_image,opacity 0.303129 151 203 65 165 opacity 0.495628 39 95 29 123
a43e755ee84f_image,None 1 0 0 1 1
c13357e59083_image,None 1 0 0 1 1
2b9cfe857b00_image,opacity 0.339572 22 90 88 214 opacity 0.469137 109 209 75 207
d6bfc693ebd0_image,None 1 0 0 1 1
7b5d82423dec_image,opacity 0.330695 146 200 66 166
38b2e04ea601_image,opacity 0.373516 44 108 1 109
94573ab529f3_image,None 1 0 0 1 1
536da2d8df21_image,None 1 0 0 1 1
516d9b3ee48c_image,None 1 0 0 1 1
f2eaa573742f_image,None 1 0 0 1 1
cb26c1aef6fb_image,None 1 0 0 1 1
e0daaf8d36a4_image,opacity 0.433172 159 215 104 240 opacity 0.54268 26 78 102 234
50de91c4297d_image,opacity 0.5343 155 223 53 183 opacity 0.577954 42 116 60 192
55ecad406759_image,opacity 0.566044 151 203 81 201
f5b45b497add_image,opacity 0.500092 156 222 58 178 opacity 0.572845 38 106 57 165
c6c9bf98487a_image,opacity 0.490357 53 113 45 121
180a5ea482c5_image,opacity 0.371603 33 89 74 198 opacity 0.510726 135 199 70 202
661343e25b55_image,opacity 0.461871 148 200 72 200 opacity 0.496805 27 77 61 191
89015a2258aa_image,opacity 0.389851 4 52 63 165 opacity 0.492669 135 191 50 180
b04178e642ed_image,opacity 0.612432 24 94 55 191 opacity 0.6222 130 208 46 194
16972e92e827_image,None 1 0 0 1 1
39756fea8654_image,None 1 0 0 1 1
4c3ca06916a1_image,None 1 0 0 1 1
2663588e7048_image,None 1 0 0 1 1
c1de38ec41d6_image,None 1 0 0 1 1
61884a13a399_image,opacity 0.34768 157 217 61 219
ba4b17aaafc2_image,opacity 0.338577 31 101 14 122 opacity 0.451741 164 224 29 129
c010822b2938_image,None 1 0 0 1 1
6f9d2683e93e_image,None 1 0 0 1 1
cb3eb8d1abd3_image,None 1 0 0 1 1
1c8df7067e35_image,opacity 0.295842 32 88 69 163
51aedbecbf46_image,None 1 0 0 1 1
90fa4db1ab87_image,opacity 0.330286 180 242 43 167 opacity 0.352836 46 106 42 164
db9ea9c5067e_image,opacity 0.331188 56 108 51 175 opacity 0.389657 170 232 53 149
4ea555b6e2a8_image,None 1 0 0 1 1
0c2ad4c7a6c4_image,None 1 0 0 1 1
b9ebfc00c1b5_image,opacity 0.586805 177 237 103 209
ab276c041ef9_image,None 1 0 0 1 1
eb6cabb0b9f2_image,opacity 0.320967 160 206 102 186 opacity 0.353952 60 108 104 180
14658311d76e_image,None 1 0 0 1 1
57d169dabba7_image,None 1 0 0 1 1
9a526c2b4891_image,None 1 0 0 1 1
4040afec3ee4_image,opacity 0.5206 41 107 32 182 opacity 0.537275 135 221 14 130
942313ae4303_image,None 1 0 0 1 1
ad3e33d5f844_image,None 1 0 0 1 1
5020cd97beb8_image,opacity 0.394563 43 113 100 234
9ad1e1a0f97d_image,None 1 0 0 1 1
927bd54fd2dd_image,opacity 0.516772 145 199 81 193 opacity 0.55923 57 107 59 183
697a674d44e0_image,opacity 0.345023 177 225 45 149 opacity 0.426109 5 55 44 142
d021e4e09628_image,opacity 0.43967 165 235 121 233 opacity 0.450335 38 116 108 244
1b282faf0f42_image,None 1 0 0 1 1
ff88ed1df29b_image,None 1 0 0 1 1
d0c7ebd93e70_image,None 1 0 0 1 1
1c1c48cb66e4_image,opacity 0.28637 129 183 70 146
b5b00763a191_image,None 1 0 0 1 1
700b4524a9cf_image,None 1 0 0 1 1
181a2a300bb8_image,opacity 0.52185 41 101 31 147
0562436fde50_image,opacity 0.530785 147 211 81 189 opacity 0.605613 47 107 63 187
550a6a9ee91e_image,None 1 0 0 1 1
1d981fb423e0_image,None 1 0 0 1 1
e9783e91ed84_image,opacity 0.345664 17 65 47 167 opacity 0.357099 151 199 58 158
c5c3fab94f6f_image,None 1 0 0 1 1
9e78e0ae2f3e_image,opacity 0.385099 40 110 37 165
049fce8128f9_image,opacity 0.317159 148 212 94 190 opacity 0.524207 35 111 34 166
833fc5c718dd_image,opacity 0.492281 168 236 69 227 opacity 0.519704 57 119 83 207
c61fe263d5e3_image,opacity 0.293891 161 215 68 142 opacity 0.314006 55 107 43 151
962e120703d1_image,None 1 0 0 1 1
6e13fdb97952_image,None 1 0 0 1 1
ae357fad39c2_image,opacity 0.298458 37 101 58 190
a9d318c0b89e_image,opacity 0.413056 133 217 46 200 opacity 0.589046 33 105 52 194
2250e31a4999_image,opacity 0.360405 161 219 68 168
f34e1aaea8a0_image,None 1 0 0 1 1
87d8baf120a6_image,opacity 0.399768 169 225 107 187
476dec5c36e1_image,None 1 0 0 1 1
87c51db67bf7_image,opacity 0.57089 17 79 122 238
afe360acb63b_image,opacity 0.372853 138 226 126 242
ad7c68b8f111_image,None 1 0 0 1 1
c440ed025b38_image,None 1 0 0 1 1
52f02af216bb_image,None 1 0 0 1 1
26f2478e3e85_image,None 1 0 0 1 1
e3cd1fdf2444_image,None 1 0 0 1 1
fec0249d70e4_image,None 1 0 0 1 1
237aafc42d57_image,None 1 0 0 1 1
7f49e2face0c_image,opacity 0.347182 150 194 50 170
3d623665afd7_image,None 1 0 0 1 1
32b23b347bb0_image,None 1 0 0 1 1
1eb558739c44_image,None 1 0 0 1 1
aad5bc70a20e_image,opacity 0.331479 187 231 89 199
32ecb7330e19_image,opacity 0.446289 184 240 67 175
c5b3475f3ea5_image,None 1 0 0 1 1
576cbe3e3680_image,None 1 0 0 1 1
3f133fac447c_image,None 1 0 0 1 1
65be7fc2448a_image,opacity 0.298151 165 229 104 208
b1b46fa2505e_image,opacity 0.333728 45 101 82 198 opacity 0.517997 170 230 71 205
154b6a49ff75_image,None 1 0 0 1 1
5ee582e2463a_image,opacity 0.326336 161 243 25 205 opacity 0.408827 46 118 20 148
2857f0e19e09_image,opacity 0.377857 176 240 97 193 opacity 0.579847 45 121 74 186
c1bb54cfe02f_image,None 1 0 0 1 1
e002731bb6cb_image,None 1 0 0 1 1
c3c99fea8c77_image,opacity 0.316251 50 110 95 191 opacity 0.384055 159 215 106 198
94cb9518e355_image,None 1 0 0 1 1
4ffdca4189d5_image,None 1 0 0 1 1
fbf41ac72bdb_image,opacity 0.28739 25 93 68 210
2686959fb3a5_image,None 1 0 0 1 1
9114c5437cfb_image,opacity 0.384922 40 104 48 162 opacity 0.387147 152 216 86 170
34834851ec56_image,opacity 0.653115 148 212 34 190 opacity 0.66617 34 110 32 184
cf0007295ff6_image,opacity 0.397997 56 112 63 165
18a23df4fe99_image,opacity 0.291912 23 71 60 172
f35d7c728684_image,None 1 0 0 1 1
0cc69a8f538c_image,opacity 0.514906 154 232 32 190
3fe73d3b28b9_image,None 1 0 0 1 1
394892cb8a63_image,None 1 0 0 1 1
78d62cb36b01_image,None 1 0 0 1 1
45a57e934e94_image,None 1 0 0 1 1
ae001742d48b_image,None 1 0 0 1 1
553cc2a7b556_image,opacity 0.336087 37 107 68 170
9677eff4374d_image,None 1 0 0 1 1
eb793f1aa7a1_image,None 1 0 0 1 1
09b64291b1b9_image,None 1 0 0 1 1
7934201faebb_image,None 1 0 0 1 1
24ef59e133f6_image,opacity 0.36386 148 188 44 128
4eb36901f082_image,opacity 0.352243 174 226 86 192
0d4ba5dc4a24_image,None 1 0 0 1 1
ee2b112cdd91_image,opacity 0.294554 157 213 128 200
10d3c965a5b4_image,None 1 0 0 1 1
2be1c227dbf2_image,opacity 0.33097 21 91 87 225
adc2ac4726e2_image,None 1 0 0 1 1
3b9fc021b611_image,None 1 0 0 1 1
26fa9834387e_image,opacity 0.359837 169 225 137 209
3d465a00a9b6_image,opacity 0.284634 30 72 38 130
bab3c8aa789b_image,opacity 0.467895 149 201 78 194 opacity 0.603011 52 108 55 171
98fbac3667c9_image,opacity 0.283693 33 95 53 177
fed3a460884c_image,None 1 0 0 1 1
8d6d32d0822e_image,opacity 0.521871 31 105 62 218 opacity 0.542903 139 227 56 184
0ed283a366dc_image,None 1 0 0 1 1
09a38ab5a5bd_image,None 1 0 0 1 1
89f860cec129_image,None 1 0 0 1 1
a4c570b129c3_image,opacity 0.367078 124 188 93 181 opacity 0.532232 25 93 61 171
720a67515500_image,opacity 0.367851 124 188 93 181 opacity 0.532222 25 93 61 171
f7cbc735f105_image,None 1 0 0 1 1
1308d8906c2e_image,None 1 0 0 1 1
32b211904f5b_image,opacity 0.349934 63 135 51 191 opacity 0.572468 176 240 89 197
b246eb99760f_image,opacity 0.36591 43 115 33 165 opacity 0.533516 171 235 47 171
93be428f5dcf_image,opacity 0.320872 17 87 42 174 opacity 0.36472 140 204 42 190
5b8ee5baa1d5_image,None 1 0 0 1 1
5472531dd991_image,None 1 0 0 1 1
cbb693ad2012_image,opacity 0.673616 153 221 65 147 opacity 0.697856 40 118 56 168
7b2ccb25396a_image,None 1 0 0 1 1
0f813cc26e33_image,None 1 0 0 1 1
23fd057dce2b_image,None 1 0 0 1 1
cdfdbadb85c1_image,opacity 0.486317 23 103 58 182 opacity 0.52235 140 204 89 205
366e7eb393e5_image,None 1 0 0 1 1
c39146cbda47_image,opacity 0.633006 146 206 119 243 opacity 0.659302 27 111 106 236
0c8af80ba8cb_image,None 1 0 0 1 1
d519913779f2_image,None 1 0 0 1 1
acc80ec7e327_image,None 1 0 0 1 1
5ba33931fc64_image,None 1 0 0 1 1
e47df1ad9ccf_image,opacity 0.302094 63 127 84 190 opacity 0.341314 162 222 74 208
11ef0d77a9c9_image,None 1 0 0 1 1
3b4e41cb8212_image,None 1 0 0 1 1
01f948f8e544_image,None 1 0 0 1 1
1c13336fc8a9_image,opacity 0.365874 120 198 32 174 opacity 0.666971 10 94 32 180
0821e77a60ed_image,None 1 0 0 1 1
87c38eda11ed_image,None 1 0 0 1 1
a91b4b59c145_image,opacity 0.354751 173 241 72 200 opacity 0.376082 22 90 55 211
3bacdbd6ce10_image,None 1 0 0 1 1
293014c67757_image,None 1 0 0 1 1
e987cb5ca98b_image,None 1 0 0 1 1
d4afc9da3448_image,None 1 0 0 1 1
6af96c8a94c9_image,opacity 0.29154 163 237 110 192
5163e54f5fc1_image,None 1 0 0 1 1
fd7f440d5444_image,opacity 0.532475 60 116 4 92 opacity 0.5382 158 234 4 112
16a0d37747d7_image,opacity 0.413608 54 138 41 201 opacity 0.54805 180 244 44 212
d42b21e5f6cd_image,opacity 0.349891 157 207 94 152
2dad900da56f_image,opacity 0.345943 24 72 160 256
98bab257d89b_image,opacity 0.406626 50 126 110 198
6c0e9f010b2b_image,opacity 0.387959 137 193 100 200 opacity 0.446747 41 105 92 188
f7036442740b_image,None 1 0 0 1 1
e53c7af3541a_image,None 1 0 0 1 1
b168e230f3f5_image,None 1 0 0 1 1
804538984842_image,None 1 0 0 1 1
bdd60ba8638e_image,None 1 0 0 1 1
1bc8687f938e_image,opacity 0.453675 66 114 47 167 opacity 0.45836 153 197 47 179
7984338fa964_image,opacity 0.482189 154 216 63 215 opacity 0.507105 25 75 71 221
36be4f943382_image,None 1 0 0 1 1
261b3d1adfe4_image,None 1 0 0 1 1
b30d2aef985f_image,opacity 0.393934 44 110 44 156
37bf83df1b86_image,opacity 0.379062 147 211 47 157
3dfa8d9f66f4_image,None 1 0 0 1 1
375b61c5cc94_image,opacity 0.39017 161 213 49 175 opacity 0.433981 46 90 49 129
f6ba3df9a8be_image,None 1 0 0 1 1
75b2c9f1f232_image,opacity 0.432258 22 86 79 171 opacity 0.484205 156 220 97 177
28b8c44feeb7_image,None 1 0 0 1 1
681766974456_image,None 1 0 0 1 1
f900aac4111b_image,None 1 0 0 1 1
f99f9714b911_image,opacity 0.424396 12 92 49 179 opacity 0.436586 135 199 90 186
2c1c4ab77d52_image,opacity 0.316679 59 103 100 198 opacity 0.358704 134 182 97 197
604bd55c21c6_image,opacity 0.307033 17 85 99 239 opacity 0.318406 147 199 98 156
fbb175d04c3b_image,None 1 0 0 1 1
82589da89e95_image,None 1 0 0 1 1
802b634297a0_image,opacity 0.284553 179 231 71 171 opacity 0.377473 52 114 49 171
1b4cf5e0fbe0_image,None 1 0 0 1 1
c4cea0fb5e7d_image,None 1 0 0 1 1
ef25824e7b3a_image,opacity 0.44052 156 222 59 173 opacity 0.568372 43 111 40 176
a5f2fd8c90cf_image,None 1 0 0 1 1
6ea59a2ee627_image,None 1 0 0 1 1
471f3f81a718_image,None 1 0 0 1 1
de0cf08fcad0_image,None 1 0 0 1 1
337662fe44ef_image,None 1 0 0 1 1
773998cb3cbe_image,None 1 0 0 1 1
574b38d8f6ee_image,opacity 0.375646 160 208 90 198 opacity 0.580783 46 106 76 196
fbc094e9276f_image,None 1 0 0 1 1
3bf0dc73f1fe_image,None 1 0 0 1 1
12fbfd355d88_image,None 1 0 0 1 1
8857adac8318_image,None 1 0 0 1 1
4e126f476f86_image,opacity 0.682019 54 118 41 177 opacity 0.683677 150 214 56 184
44b12000b7bf_image,None 1 0 0 1 1
a27f34f4faa2_image,None 1 0 0 1 1
731b94c566ef_image,opacity 0.309687 161 245 113 253
fb1e3b68864d_image,opacity 0.503303 31 111 75 231
d4a4dc17bb61_image,opacity 0.552728 37 109 28 188
e9a3b9253cd3_image,None 1 0 0 1 1
dcf6740dbc49_image,opacity 0.427385 61 125 130 226 opacity 0.432373 166 234 130 230
2fd67e10bfe7_image,opacity 0.537907 172 236 68 172 opacity 0.541482 72 134 57 177
4855b78e0df2_image,None 1 0 0 1 1
11dbf622932a_image,None 1 0 0 1 1
2ae9c4c6566f_image,None 1 0 0 1 1
cf0bf9859b12_image,opacity 0.505336 149 203 59 187
5a73ab51749e_image,None 1 0 0 1 1
cc2c3a3d2f4b_image,None 1 0 0 1 1
a2bc78e322a8_image,None 1 0 0 1 1
2a2f54eb48df_image,None 1 0 0 1 1
5badf765648c_image,None 1 0 0 1 1
95691f6666a0_image,None 1 0 0 1 1
5fba37ac7937_image,opacity 0.326869 149 197 47 151
57879f49e8a7_image,None 1 0 0 1 1
b138b32605b0_image,None 1 0 0 1 1
ca107bb3a592_image,opacity 0.497042 147 215 84 196
e49489fcf5c0_image,opacity 0.50378 149 217 102 202
b3c8408f4064_image,None 1 0 0 1 1
ecb545d2ff15_image,None 1 0 0 1 1
30b1c089724e_image,opacity 0.371256 129 183 62 168 opacity 0.477298 32 96 56 160
8262b27852b3_image,opacity 0.546267 9 83 30 154 opacity 0.590234 125 191 28 182
5b75a1de503b_image,opacity 0.44608 19 103 3 131 opacity 0.593413 140 204 14 162
bc72a19f5d87_image,None 1 0 0 1 1
ab15692269fc_image,None 1 0 0 1 1
376d6d181b25_image,opacity 0.615514 161 253 91 249 opacity 0.61566 60 116 102 242
5378fb58d2d1_image,opacity 0.294897 150 206 75 147 opacity 0.529227 48 112 74 192
946a0aa2bb58_image,opacity 0.289145 32 104 56 216
1199fd9f2aba_image,None 1 0 0 1 1
a459429226e0_image,opacity 0.36424 144 216 111 191 opacity 0.4009 37 101 138 206
8bc924bcab6f_image,opacity 0.37068 148 196 106 174
d16be9684636_image,opacity 0.323917 54 122 65 211
80af9d802a16_image,None 1 0 0 1 1
6f3931d7257e_image,None 1 0 0 1 1
859cac874454_image,None 1 0 0 1 1
6c45d9b8e386_image,opacity 0.309264 33 89 154 206
8c3864112d54_image,opacity 0.450871 161 217 167 235
767f10b9aab5_image,opacity 0.289689 172 232 63 173 opacity 0.495724 40 112 56 200
7fb9c360a3a0_image,None 1 0 0 1 1
3dcdfc352a06_image,None 1 0 0 1 1
609361f18122_image,opacity 0.371402 63 119 113 193 opacity 0.37191 151 197 130 194
978e985f0c70_image,None 1 0 0 1 1
3be5852db61e_image,opacity 0.308447 67 119 55 111
db1af486b119_image,None 1 0 0 1 1
c6a555431645_image,opacity 0.64634 150 214 54 172 opacity 0.651808 43 111 52 172
a4067adac314_image,None 1 0 0 1 1
5232ab4c4671_image,None 1 0 0 1 1
18e043b70fda_image,None 1 0 0 1 1
47ea205fd755_image,None 1 0 0 1 1
603addd70746_image,opacity 0.296519 58 110 98 152 opacity 0.362401 159 207 60 164
51ea4e1227bd_image,None 1 0 0 1 1
ae9849096225_image,opacity 0.314732 19 71 48 140
8b6ebd55c8c8_image,opacity 0.37241 57 105 70 168 opacity 0.383507 155 203 96 174
b9f9619f02ba_image,opacity 0.305319 177 233 89 221
a0f473f71878_image,opacity 0.376157 192 256 2 142 opacity 0.600812 78 162 3 115
ede662a3de87_image,opacity 0.33782 143 215 84 212 opacity 0.592593 5 101 69 217
5175b9a4a25e_image,None 1 0 0 1 1
16237f3bc333_image,opacity 0.322163 173 221 98 212 opacity 0.349917 32 88 120 192
4f7f40e478b1_image,None 1 0 0 1 1
08992bd07010_image,opacity 0.374166 38 92 65 165 opacity 0.45076 152 208 92 178
b69c808bc09b_image,None 1 0 0 1 1
53c971120ab1_image,opacity 0.551741 29 103 54 178
3d4fb4320278_image,None 1 0 0 1 1
b873cfe5cead_image,None 1 0 0 1 1
ba0b49821b69_image,None 1 0 0 1 1
114f7f3baedf_image,opacity 0.378898 39 93 54 180 opacity 0.385754 140 196 44 196
82198019c4a4_image,opacity 0.494671 181 255 97 181
1b640a2839db_image,None 1 0 0 1 1
d5911a060ee4_image,opacity 0.478763 50 110 65 211 opacity 0.604373 158 222 65 213
4fff9ae1ff37_image,None 1 0 0 1 1
e8cc6a2f117c_image,opacity 0.433469 156 212 56 168
93dd10a00c51_image,None 1 0 0 1 1
bf30cc575dc9_image,opacity 0.297114 164 210 79 167 opacity 0.488671 59 103 70 146
11541455ef19_image,opacity 0.544905 49 121 10 118 opacity 0.582655 168 232 7 137
4905b651a091_image,opacity 0.28346 60 116 94 138
97998832742c_image,opacity 0.432349 39 103 52 204 opacity 0.486781 155 213 56 210
95be2813ba36_image,opacity 0.593396 145 241 66 208 opacity 0.639798 36 116 68 204
17e40efa4596_image,None 1 0 0 1 1
198ffac65dac_image,None 1 0 0 1 1
d5d3ededa34e_image,opacity 0.366578 35 87 76 158 opacity 0.515718 146 210 58 172
6fd59d569aa1_image,None 1 0 0 1 1
af31df978fe9_image,None 1 0 0 1 1
6c42a41a6c6e_image,None 1 0 0 1 1
6eb71d191dec_image,None 1 0 0 1 1
36500eabad2d_image,None 1 0 0 1 1
05feea9719f5_image,opacity 0.549063 134 210 4 138 opacity 0.554349 17 93 6 128
ad97e89134ca_image,opacity 0.351592 185 241 63 175
a489286e78c1_image,opacity 0.309843 180 242 133 211
1c1e9d3f5bab_image,None 1 0 0 1 1
5e64da034f7a_image,opacity 0.571515 177 229 100 176
9d0f2c2645da_image,opacity 0.571515 177 229 100 176
ea9ee4d59c59_image,opacity 0.571515 177 229 100 176
8032d270ede6_image,None 1 0 0 1 1
d2289e68cca1_image,opacity 0.343848 159 211 96 176 opacity 0.470618 69 117 49 129
a24bc9ab2ea8_image,opacity 0.303577 130 212 49 177 opacity 0.592406 33 101 49 189
e134e2ffdfaf_image,None 1 0 0 1 1
91a36aae8b27_image,None 1 0 0 1 1
7d0b75feaa8a_image,None 1 0 0 1 1
268d472480ec_image,opacity 0.351028 39 99 58 222 opacity 0.369309 161 221 50 216
e3eeddbfba24_image,opacity 0.513027 32 104 27 171 opacity 0.595664 143 207 26 158
1a4c378605f5_image,None 1 0 0 1 1
bb6fc945fc18_image,opacity 0.315717 30 108 65 217 opacity 0.428179 166 250 75 211
159cbf4e41e4_image,None 1 0 0 1 1
a591e663f4d8_image,None 1 0 0 1 1
9f525be80c70_image,opacity 0.345721 3 71 99 195
a82ca8f37fb6_image,opacity 0.361066 150 202 80 164 opacity 0.364785 53 105 50 150
ffc66893d9ea_image,None 1 0 0 1 1
9883f5325492_image,opacity 0.429933 64 116 100 244 opacity 0.47457 171 219 87 223
ad56b89a81c7_image,opacity 0.558923 52 124 61 189 opacity 0.604487 159 239 63 197
da263425b379_image,None 1 0 0 1 1
070cbe2d3dbb_image,None 1 0 0 1 1
d8b8ef2bcbf6_image,None 1 0 0 1 1
14ec6bdb5641_image,opacity 0.31792 16 68 58 166
6d66127d071d_image,opacity 0.285668 139 191 68 164
db15ead7ab00_image,None 1 0 0 1 1
71dfba1a08de_image,None 1 0 0 1 1
a22e6a0c0ec3_image,None 1 0 0 1 1
b39bb8f504fb_image,opacity 0.449642 21 71 36 114 opacity 0.48101 144 200 42 118
e5a860104190_image,None 1 0 0 1 1
100316af8a36_image,None 1 0 0 1 1
d6387e1c8161_image,None 1 0 0 1 1
5c88a5fc55e7_image,opacity 0.462091 46 106 49 173 opacity 0.471848 162 222 63 167
2743e121b370_image,None 1 0 0 1 1
cea591e99b8a_image,opacity 0.317635 173 221 105 197
3d639e7999a5_image,opacity 0.320628 71 141 127 189 opacity 0.404845 175 243 73 159
aebc1d3a968d_image,None 1 0 0 1 1
d0439f9e0dcd_image,None 1 0 0 1 1
0be236941397_image,None 1 0 0 1 1
d6809af17965_image,None 1 0 0 1 1
4b0058c1f54e_image,opacity 0.337935 166 224 116 244 opacity 0.477151 14 74 127 239
0a3a99612b28_image,opacity 0.650191 50 100 47 169
639f6bf3c727_image,None 1 0 0 1 1
f554169c1277_image,opacity 0.525719 44 112 45 161 opacity 0.566475 164 220 44 164
5a0de0207028_image,None 1 0 0 1 1
5596c4063d58_image,None 1 0 0 1 1
539931eced05_image,opacity 0.538759 16 92 36 164 opacity 0.54695 137 203 38 170
c8538fe73673_image,opacity 0.458732 135 231 43 187 opacity 0.616353 5 109 25 161
877b017a6d20_image,None 1 0 0 1 1
7230f166720e_image,opacity 0.555782 59 123 84 162 opacity 0.655721 159 211 85 181
4f1ad3374103_image,None 1 0 0 1 1
41c26ea10186_image,None 1 0 0 1 1
cfcbddf07c58_image,None 1 0 0 1 1
11b60db4052a_image,None 1 0 0 1 1
3d56a237b6be_image,None 1 0 0 1 1
69ace2980660_image,None 1 0 0 1 1
5e0e7acd9c7d_image,opacity 0.312169 156 250 19 131
6ca497c845ef_image,opacity 0.453593 139 201 178 248
efcfaa82fcb8_image,None 1 0 0 1 1
e92b08b5a77c_image,None 1 0 0 1 1
ca7b91d018c7_image,opacity 0.357037 189 249 94 178 opacity 0.407594 82 144 64 168
9f6cc74b0062_image,None 1 0 0 1 1
5878b290f09a_image,opacity 0.299402 136 198 176 240 opacity 0.373208 7 71 181 241
fd435a3c012d_image,opacity 0.354968 179 233 48 166 opacity 0.394376 54 100 59 181
bb1d9e39b848_image,opacity 0.488457 158 210 33 157 opacity 0.654601 52 108 31 137
f2200b845e93_image,opacity 0.316495 161 225 52 180
0fedf74dc163_image,opacity 0.333666 178 248 30 202 opacity 0.511466 39 143 15 199
7cc2f3d9b469_image,opacity 0.308426 40 96 20 156 opacity 0.446466 144 200 27 155
386343f0df08_image,opacity 0.476663 34 98 94 184 opacity 0.518951 180 236 92 196
f533f5259633_image,opacity 0.306374 148 218 29 161 opacity 0.461305 27 111 24 166
4c2d931d0f33_image,opacity 0.641529 6 90 79 215 opacity 0.668354 135 203 81 221
256f0960de75_image,None 1 0 0 1 1
e5f1d33714e6_image,opacity 0.434283 163 227 26 158 opacity 0.573014 68 118 36 158
c34113e0228a_image,None 1 0 0 1 1
1e1734034b0f_image,None 1 0 0 1 1
0026720152f5_image,None 1 0 0 1 1
949b8bf1079a_image,opacity 0.299679 151 231 71 211 opacity 0.327705 47 115 56 208
d69b4547fc69_image,None 1 0 0 1 1
ecdd379e2a9b_image,opacity 0.464028 146 226 158 250
b22a9e82f34e_image,None 1 0 0 1 1
e2f83c4e9c79_image,None 1 0 0 1 1
0db323463d0a_image,None 1 0 0 1 1
eff9f15c7e9b_image,opacity 0.367918 43 107 135 187
664250f0f623_image,None 1 0 0 1 1
1aa1491c9cd1_image,None 1 0 0 1 1
1093f305a358_image,opacity 0.391186 18 70 65 149
33087aa9853d_image,None 1 0 0 1 1
eebfaa5e1a65_image,opacity 0.354075 43 103 82 206
83594e6ad0ba_image,None 1 0 0 1 1
c0c89f864fda_image,None 1 0 0 1 1
37034ce54151_image,None 1 0 0 1 1
50537bcf4cc2_image,None 1 0 0 1 1
5f65421ff6fd_image,None 1 0 0 1 1
2ddefbb957e9_image,None 1 0 0 1 1
160cfef3014f_image,opacity 0.350784 174 218 88 192
da3860170bad_image,opacity 0.320529 64 120 89 181
349d5e5d8690_image,None 1 0 0 1 1
8d2c75ff8149_image,None 1 0 0 1 1
3612ee150596_image,None 1 0 0 1 1
0feac35cd28b_image,opacity 0.33189 6 50 67 141
51635cbfbe18_image,opacity 0.365991 28 92 91 163
0f50aabaa314_image,None 1 0 0 1 1
b717534f157a_image,None 1 0 0 1 1
d3b7a21b2ce2_image,None 1 0 0 1 1
39f8e38c9d43_image,opacity 0.448744 178 230 72 216
e730aa7fca0a_image,opacity 0.351335 45 101 24 136 opacity 0.396741 147 201 35 131
be53bfb563b3_image,opacity 0.400548 149 213 72 214 opacity 0.426644 25 97 74 198
1dccbb75ffba_image,None 1 0 0 1 1
c7ef36caa2fa_image,None 1 0 0 1 1
a29c5a68b07b_image,opacity 0.519034 59 131 77 205 opacity 0.558362 176 248 83 201
ef661638c8c0_image,opacity 0.573159 158 210 48 174 opacity 0.583616 45 105 46 178
776686ca2f1b_image,None 1 0 0 1 1
7603bfc07ffd_image,None 1 0 0 1 1
5dd9e477a6ee_image,opacity 0.347886 37 89 46 178
2a316bf62f59_image,opacity 0.345474 59 105 126 192
5dba2b3647b1_image,opacity 0.405956 139 195 49 167 opacity 0.568315 44 100 50 166
7169a450d1bf_image,None 1 0 0 1 1
88b9ca325a4c_image,None 1 0 0 1 1
b4a7026fe17c_image,None 1 0 0 1 1
ee580bf07fa1_image,None 1 0 0 1 1
c788095df8f2_image,opacity 0.404059 41 105 60 194 opacity 0.418926 180 244 71 195
d764b622b4b5_image,opacity 0.432908 143 203 123 221 opacity 0.437912 25 81 70 216
699fc1ec26f9_image,opacity 0.392828 145 193 66 150 opacity 0.501857 62 108 59 147
ac634dd7e884_image,None 1 0 0 1 1
7230234e120a_image,opacity 0.328651 34 112 97 181 opacity 0.407256 156 234 128 216
6be63db536d7_image,opacity 0.429363 146 198 99 219 opacity 0.631907 43 109 99 219
4c3dbddfc294_image,None 1 0 0 1 1
65045f5ef280_image,None 1 0 0 1 1
2fca6fe9c1ba_image,None 1 0 0 1 1
b708990b8e03_image,opacity 0.643319 54 112 63 167 opacity 0.678265 147 205 68 156
6fd7971538df_image,None 1 0 0 1 1
84d25a1cef94_image,None 1 0 0 1 1
374e1b87d7b8_image,opacity 0.475376 19 115 45 153 opacity 0.497541 178 246 57 159
087f2e308700_image,None 1 0 0 1 1
9951f793880c_image,None 1 0 0 1 1
c955b7274092_image,None 1 0 0 1 1
be4b4301c5e4_image,opacity 0.387019 20 88 72 192 opacity 0.439133 145 201 63 195
6828795ea830_image,opacity 0.399146 142 194 81 201 opacity 0.428618 28 92 54 162
89426c0c18a8_image,None 1 0 0 1 1
2be224a00fac_image,opacity 0.533821 139 211 54 210 opacity 0.615627 2 94 42 188
18e6a4083920_image,None 1 0 0 1 1
21ab319a926c_image,opacity 0.601483 17 77 42 180
3730eb76afba_image,opacity 0.307953 33 89 98 198
8ad2e769d2b0_image,None 1 0 0 1 1
d0de673a7b63_image,None 1 0 0 1 1
cdd9c6e8d821_image,None 1 0 0 1 1
503a6e0884c6_image,opacity 0.326344 35 105 132 208
4fe79f658041_image,None 1 0 0 1 1
fc45a3b144a7_image,None 1 0 0 1 1
25d33047633c_image,opacity 0.382514 143 207 89 215 opacity 0.519515 30 104 62 224
3d47cc325262_image,None 1 0 0 1 1
1ed201489300_image,None 1 0 0 1 1
5c4c752c1e46_image,None 1 0 0 1 1
af13d77fc159_image,opacity 0.516196 50 118 79 219 opacity 0.613719 163 235 70 216
2b59777ebc31_image,opacity 0.505081 162 230 40 170 opacity 0.581696 48 112 32 144
3b7ae6879147_image,None 1 0 0 1 1
4fc12ae446a8_image,opacity 0.458253 177 233 37 169 opacity 0.536288 54 138 10 142
2a2019322d8e_image,None 1 0 0 1 1
de56c2d5b778_image,None 1 0 0 1 1
9d2e46f17c72_image,None 1 0 0 1 1
1e9b7b493e47_image,opacity 0.419821 48 104 47 175 opacity 0.455993 171 231 85 189
9f10e1534e81_image,None 1 0 0 1 1
6de46911f968_image,None 1 0 0 1 1
f90deb13b3e2_image,opacity 0.461872 48 96 71 155
2565c5ccfba1_image,opacity 0.341408 163 221 115 179 opacity 0.514993 32 104 99 173
6dea3c3d1efa_image,opacity 0.281047 34 102 47 115 opacity 0.412996 156 222 30 134
ddd7ad8b0c41_image,opacity 0.55765 24 88 72 192 opacity 0.576045 134 210 67 195
1eae0890a6ad_image,opacity 0.29132 41 107 15 157 opacity 0.559662 164 228 60 172
3247432330d8_image,None 1 0 0 1 1
7c95cf673952_image,opacity 0.396411 158 234 66 206 opacity 0.470195 41 117 78 216
00fc8fc35dc1_image,opacity 0.512868 171 251 52 156 opacity 0.542648 47 119 35 147
82b36d045a12_image,None 1 0 0 1 1
130aa9d3716d_image,opacity 0.409874 149 211 75 125
f2038cb67c83_image,opacity 0.467774 68 132 75 235 opacity 0.665887 170 250 72 208
0f147ee4a61b_image,opacity 0.381488 179 233 82 168 opacity 0.4403 65 117 84 164
572d5c14fc66_image,opacity 0.452447 55 119 75 209 opacity 0.54388 170 238 79 211
8727f101e9f6_image,None 1 0 0 1 1
28d0fa664e61_image,opacity 0.543027 160 216 64 190 opacity 0.58959 49 117 58 214
509c535a3dc1_image,opacity 0.516746 168 248 91 215
4bf0dc7bae40_image,None 1 0 0 1 1
262f3525e16a_image,opacity 0.4796 16 114 53 213 opacity 0.533099 163 231 85 217
d1a0f21a848f_image,opacity 0.559607 172 248 47 195 opacity 0.588581 53 133 51 195
f58258eeaa42_image,None 1 0 0 1 1
16d5b63ac279_image,None 1 0 0 1 1
435bc0fcb0ab_image,None 1 0 0 1 1
accc3b15a2d4_image,opacity 0.576275 140 204 91 171
064b37b01cd2_image,None 1 0 0 1 1
5c90beaa5c48_image,opacity 0.379133 178 220 21 97 opacity 0.506215 58 110 43 131
59911e31f963_image,opacity 0.509086 168 256 79 223
a840d15a7cdc_image,opacity 0.455057 39 127 27 189 opacity 0.661481 155 255 13 161
d8e2ec854be4_image,opacity 0.448679 162 214 64 172 opacity 0.623763 48 104 43 163
dba0caa176ce_image,None 1 0 0 1 1
41c25ec8c44c_image,None 1 0 0 1 1
69d1134dccc6_image,None 1 0 0 1 1
a67ef603134f_image,opacity 0.300291 192 240 68 172 opacity 0.372389 52 102 70 178
120ce3e7cd73_image,None 1 0 0 1 1
8c452e8e1278_image,opacity 0.619697 149 229 100 252 opacity 0.672335 35 119 108 244
73eb8898d211_image,None 1 0 0 1 1
a706ebbc447a_image,opacity 0.450707 133 231 28 204
e23027c8bea6_image,None 1 0 0 1 1
aba653aebd55_image,None 1 0 0 1 1
09443dcb865f_image,None 1 0 0 1 1
c5ac37cc6234_image,None 1 0 0 1 1
9b8610d78de1_image,opacity 0.313215 12 100 141 235
f31bb833e467_image,opacity 0.380357 152 216 112 224 opacity 0.403421 35 107 63 227
dee71f5c5b9c_image,opacity 0.504614 21 89 57 161 opacity 0.560168 157 221 71 175
695e2c6dede4_image,None 1 0 0 1 1
8d6dea06a032_image,opacity 0.283769 149 209 75 183
0d47ef3e87e4_image,opacity 0.365856 165 215 90 182
0bbb78f96f46_image,None 1 0 0 1 1
dc0b4b3495c4_image,opacity 0.286618 130 182 106 174 opacity 0.383763 21 69 76 160
86734b7cc20f_image,opacity 0.353671 180 230 121 199 opacity 0.39107 53 115 99 195
a6a0ba6efa97_image,opacity 0.486655 147 207 60 164 opacity 0.654711 39 103 28 180
53b0dc275cb0_image,opacity 0.305651 171 251 40 186 opacity 0.457187 66 134 42 182
6b07294c7674_image,None 1 0 0 1 1
d50e6d47864b_image,opacity 0.425728 19 99 47 181
aa6b768d3161_image,opacity 0.506188 50 126 40 158 opacity 0.577166 167 231 67 179
ea71a8b95d30_image,opacity 0.505891 50 126 40 158 opacity 0.57704 167 231 67 179
cee98095dd28_image,opacity 0.419166 16 112 48 194 opacity 0.460352 175 247 77 209
d1be32513936_image,opacity 0.289388 15 83 58 162
ac2562a40897_image,None 1 0 0 1 1
9310ceec556a_image,None 1 0 0 1 1
55f69f03ee2f_image,opacity 0.317223 180 228 129 189
92c25d90d0a8_image,opacity 0.388124 60 110 147 203
8927235865c9_image,opacity 0.281368 171 227 62 154
51d2f37f3f60_image,None 1 0 0 1 1
1b40a20045ee_image,None 1 0 0 1 1
19e9733ee506_image,None 1 0 0 1 1
ec563919514c_image,None 1 0 0 1 1
9c297e900f67_image,None 1 0 0 1 1
f785f9c6bbf7_image,opacity 0.478811 127 227 26 188 opacity 0.4968 27 103 26 158
275513358c80_image,None 1 0 0 1 1
cf292f843e25_image,None 1 0 0 1 1
cad5ad977e21_image,None 1 0 0 1 1
e59126042b0e_image,opacity 0.290386 55 115 49 169 opacity 0.341099 177 225 52 172
07eb8cf8958a_image,None 1 0 0 1 1
1f927318863d_image,None 1 0 0 1 1
5592c8778ae2_image,None 1 0 0 1 1
bef0837ba654_image,None 1 0 0 1 1
158cc858a485_image,opacity 0.333741 165 225 53 175 opacity 0.468485 39 101 45 195
8829d22c3e2b_image,opacity 0.518669 145 191 62 170
98118511c57b_image,opacity 0.326034 50 94 49 177 opacity 0.474752 170 224 51 175
4c4737c3e3f8_image,None 1 0 0 1 1
040d0765b661_image,None 1 0 0 1 1
649b529933d6_image,opacity 0.325134 15 63 85 213 opacity 0.368625 149 201 108 204
ae9c9f0b03da_image,opacity 0.48234 38 110 55 223
d6c2f07ca4ba_image,opacity 0.417479 58 110 141 205
9aa2c2119c43_image,opacity 0.446099 47 107 25 145
37cbf79c6de0_image,None 1 0 0 1 1
b1c455a826eb_image,opacity 0.28476 140 188 67 173 opacity 0.424388 61 117 77 197
fbf51388c4ff_image,None 1 0 0 1 1
f625309a6d62_image,None 1 0 0 1 1
fbdb5bff26f2_image,None 1 0 0 1 1
6a0889223a38_image,None 1 0 0 1 1
d62e3d8adae9_image,opacity 0.625968 23 79 63 205
904324209389_image,None 1 0 0 1 1
fe97de4edad5_image,None 1 0 0 1 1
1078af8ac3b7_image,None 1 0 0 1 1
405d774f17d5_image,None 1 0 0 1 1
b09b20b44640_image,None 1 0 0 1 1
61505043bc32_image,None 1 0 0 1 1
d25061b50086_image,opacity 0.529476 143 207 101 251 opacity 0.559768 23 83 80 196
05a70a1c16c1_image,opacity 0.35748 40 104 51 191 opacity 0.583581 149 201 81 213
95b2cd351ef0_image,None 1 0 0 1 1
866f58f2ff1f_image,None 1 0 0 1 1
c75d657e8065_image,None 1 0 0 1 1
5aa20bb0feb2_image,None 1 0 0 1 1
2533fc9ec975_image,None 1 0 0 1 1
e8d690e0d67d_image,None 1 0 0 1 1
047ba804182b_image,None 1 0 0 1 1
2e4e44dfe103_image,None 1 0 0 1 1
1ade3b5d0699_image,opacity 0.579395 162 238 74 214 opacity 0.684251 26 118 68 222
1d9a8f9f0eef_image,opacity 0.537674 149 213 57 177 opacity 0.56677 23 99 48 178
35271bec39b5_image,opacity 0.347991 154 206 46 154 opacity 0.36655 57 105 33 149
551957feafcc_image,opacity 0.338528 151 213 130 214
8b7e42d19053_image,opacity 0.506658 127 187 66 182 opacity 0.61682 6 80 60 170
52168ec4f7c5_image,None 1 0 0 1 1
3550632514dc_image,opacity 0.531201 156 220 75 167 opacity 0.546135 60 120 62 186
96e1e1cf5431_image,opacity 0.400908 159 209 50 160 opacity 0.585989 53 103 26 158
6a99f80ffa1b_image,None 1 0 0 1 1
f49845505871_image,opacity 0.312504 62 106 62 178 opacity 0.509739 184 242 82 214
04396dd5af2f_image,opacity 0.491476 142 210 89 197
e036a057d0cf_image,None 1 0 0 1 1
3f8195c5362d_image,None 1 0 0 1 1
290d07d7480f_image,opacity 0.336524 164 220 50 186
b5f9a65597f2_image,None 1 0 0 1 1
d6af3acc4ce6_image,None 1 0 0 1 1
4810ae5dbc08_image,opacity 0.410654 147 193 24 184 opacity 0.575598 36 96 9 141
dbae9b9b9500_image,None 1 0 0 1 1
a1883aeaa8f0_image,opacity 0.309993 19 79 132 196
6746a1932779_image,None 1 0 0 1 1
4d1bf3193a0c_image,None 1 0 0 1 1
6cc1f1900818_image,None 1 0 0 1 1
b16a04496b57_image,opacity 0.397179 53 115 39 117 opacity 0.622819 167 231 22 138
2f9ccb6faa68_image,None 1 0 0 1 1
236d240402cd_image,opacity 0.579221 73 137 61 205 opacity 0.592169 188 246 52 182
49fb3141efe3_image,opacity 0.549289 37 97 40 136 opacity 0.579093 151 219 44 156
6bce1f945580_image,None 1 0 0 1 1
8fbc90b33f87_image,opacity 0.529077 161 215 52 170 opacity 0.556819 49 109 47 167
e19d8286b13b_image,opacity 0.487821 175 231 126 218 opacity 0.657572 64 136 94 210
34b713a5bb1d_image,opacity 0.284454 46 114 49 155 opacity 0.328199 146 198 91 191
d706eea1b70c_image,None 1 0 0 1 1
c3d79bd7e833_image,opacity 0.302526 70 130 65 235 opacity 0.576407 169 233 66 214
ed74d42fc445_image,opacity 0.515871 56 112 89 177
222601cefe1b_image,opacity 0.387286 191 253 41 185 opacity 0.555105 91 151 27 151
acc3e6eff7c8_image,None 1 0 0 1 1
81ca75341b65_image,opacity 0.311851 44 100 53 161
5c9f41283e7c_image,None 1 0 0 1 1
49e9379c42dc_image,opacity 0.579538 153 225 34 178 opacity 0.611071 59 119 35 155
275c11b52bec_image,None 1 0 0 1 1
bea59325e32c_image,None 1 0 0 1 1
b5b53f0a4947_image,opacity 0.337431 144 194 141 219 opacity 0.37824 51 103 90 198
9b7cee40d0a7_image,opacity 0.43533 150 214 90 242 opacity 0.547964 21 105 76 220
0f01c0517022_image,opacity 0.293984 58 112 49 117 opacity 0.466125 167 215 52 172 opacity 0.55288 52 108 49 177
8b354d4a216f_image,opacity 0.301026 48 98 77 167 opacity 0.434173 170 222 55 179
3a81bc9c91c9_image,None 1 0 0 1 1
ced40f593496_image,None 1 0 0 1 1
c8ebcc491f27_image,None 1 0 0 1 1
34270243e572_image,None 1 0 0 1 1
d09b2158a7c3_image,None 1 0 0 1 1
6c4cdd180e0d_image,None 1 0 0 1 1
a3d2fd5ef626_image,None 1 0 0 1 1
9603112b4c2e_image,opacity 0.360833 148 196 81 169 opacity 0.533059 33 97 39 175
0718c860f2ee_image,None 1 0 0 1 1
b3d87adb933b_image,opacity 0.446435 129 181 93 197
4eceeddd00ca_image,opacity 0.333385 22 94 67 203 opacity 0.40514 134 226 71 215
c614d8c2e47e_image,opacity 0.376916 6 74 60 202
c0f5131b1c99_image,opacity 0.298276 46 110 81 165 opacity 0.544204 180 232 64 168
6ca7b75a4796_image,None 1 0 0 1 1
bb6f8089e6df_image,None 1 0 0 1 1
25e11b9fb687_image,None 1 0 0 1 1
dfb6a487e854_image,opacity 0.548468 26 90 56 224
a52729755130_image,opacity 0.446028 8 64 98 200
7f30a3c92da5_image,None 1 0 0 1 1
1f737bcf6fe9_image,None 1 0 0 1 1
50243b2cdb56_image,opacity 0.360877 44 100 120 176
5ac874337f82_image,None 1 0 0 1 1
af7d1bd1d629_image,None 1 0 0 1 1
c81d1d4a4d70_image,None 1 0 0 1 1
d3d02532e1cf_image,opacity 0.43 36 116 45 201 opacity 0.531963 162 244 33 195
cf2bb9880db5_image,None 1 0 0 1 1
df829b7602ce_image,None 1 0 0 1 1
ebaebf6b1e02_image,opacity 0.355899 17 101 17 159 opacity 0.358053 137 215 40 160
eb632adfa113_image,opacity 0.288133 79 133 50 166
340c9cf3b19a_image,None 1 0 0 1 1
4ece84b1cfb8_image,opacity 0.2964 35 111 127 219
2a1dae5f6b5f_image,None 1 0 0 1 1
b576b0f1eeac_image,None 1 0 0 1 1
eb4d5d8af5a5_image,None 1 0 0 1 1
ce34bdbe98b5_image,None 1 0 0 1 1
7c77f9aa3a25_image,None 1 0 0 1 1
cb75251076c7_image,opacity 0.304643 96 144 65 161
fd65cc14da23_image,opacity 0.514661 125 193 42 188 opacity 0.668593 16 96 30 186
c6fa6b9b8093_image,None 1 0 0 1 1
e5e44940be7a_image,None 1 0 0 1 1
3a2aa71a792e_image,None 1 0 0 1 1
a269f69b335e_image,None 1 0 0 1 1
63173909bed5_image,opacity 0.503093 43 101 54 186 opacity 0.548584 148 214 55 199
977a20a10137_image,opacity 0.316672 139 187 58 180 opacity 0.390991 54 106 54 146
d8407edd456b_image,opacity 0.396983 69 131 57 173 opacity 0.482868 178 238 68 198
13a28f6783b2_image,opacity 0.531435 49 129 44 178
1a330ec81ee3_image,opacity 0.365687 35 109 49 197
1ce9abb861b6_image,opacity 0.398866 14 62 47 185
957550f711ee_image,None 1 0 0 1 1
eeff73683974_image,None 1 0 0 1 1
9e525f16490a_image,opacity 0.322679 11 85 60 206
5eb97e03c468_image,opacity 0.43108 30 74 54 162
7a31d8763221_image,opacity 0.478154 48 128 61 217 opacity 0.484113 169 225 68 212
0af6a45668be_image,None 1 0 0 1 1
12ae063a0dda_image,opacity 0.435277 19 81 103 203
d916d2169020_image,None 1 0 0 1 1
520a1a7b039b_image,None 1 0 0 1 1
2ba135370ac3_image,opacity 0.304445 57 101 37 121 opacity 0.317183 149 197 50 174
dded54f9de15_image,None 1 0 0 1 1
0d50bce86451_image,opacity 0.539371 27 117 85 213 opacity 0.613879 167 251 90 254
ea1c164158da_image,None 1 0 0 1 1
b83b916b6714_image,opacity 0.425386 156 212 49 179
26454c05ce6e_image,None 1 0 0 1 1
797fe7acf67b_image,opacity 0.293655 46 94 69 181 opacity 0.385933 177 221 61 161
06abdb2562ce_image,None 1 0 0 1 1
4d7b06512cc1_image,opacity 0.413767 180 230 50 186
eb3fe9eb8e39_image,opacity 0.360666 163 223 114 166
60f94703051a_image,None 1 0 0 1 1
0c4d32f6b806_image,None 1 0 0 1 1
f87f1ab5729e_image,None 1 0 0 1 1
4e01dd0a5fdf_image,opacity 0.5802 153 209 72 192 opacity 0.651301 37 115 53 173
1c9c9f68941e_image,opacity 0.369017 137 197 68 168
042979d394a1_image,opacity 0.401678 154 212 39 165
7035dc2026f0_image,opacity 0.458398 169 225 76 158 opacity 0.604429 40 112 34 142
7205313f062a_image,None 1 0 0 1 1
c0a746ef9f9f_image,opacity 0.574593 47 111 58 152 opacity 0.586945 158 216 56 156
bf8e59f80d5c_image,opacity 0.549303 48 112 70 190 opacity 0.596296 163 223 78 208
84dd9eff2ecf_image,None 1 0 0 1 1
2a31b1526915_image,None 1 0 0 1 1
3ed35204df56_image,None 1 0 0 1 1
a74e51891aa8_image,opacity 0.359541 89 141 56 166 opacity 0.425705 186 238 55 179
f39044d6c511_image,None 1 0 0 1 1
b3e31b3aeba4_image,None 1 0 0 1 1
5b12f6b2c93e_image,opacity 0.398117 162 214 117 195
9a1185f5ff34_image,opacity 0.434849 144 210 114 226
82e1a7088c64_image,None 1 0 0 1 1
9bce1b51617d_image,None 1 0 0 1 1
817301ec71da_image,None 1 0 0 1 1
eaad41dd9fe3_image,opacity 0.462753 15 97 53 201 opacity 0.565487 136 192 53 187
7b0e22a045ef_image,None 1 0 0 1 1
2bc3736cc7bf_image,opacity 0.290992 40 106 60 212 opacity 0.477171 151 231 68 218
d91b58caa74d_image,None 1 0 0 1 1
818065156194_image,opacity 0.30404 38 90 139 207 opacity 0.376026 153 205 96 200
5c8c441a0650_image,None 1 0 0 1 1
d0b546b42023_image,opacity 0.328997 25 75 109 229
046b1edf6801_image,None 1 0 0 1 1
9f526498d12b_image,None 1 0 0 1 1
05435da60872_image,opacity 0.437958 35 91 70 170 opacity 0.455962 147 205 80 200
82d09edcbfc4_image,opacity 0.313044 175 233 81 153 opacity 0.340782 58 118 73 157
822b6007287e_image,None 1 0 0 1 1
045c1e859dfa_image,None 1 0 0 1 1
1c937e2b91a9_image,opacity 0.453887 37 101 68 164 opacity 0.456007 165 237 95 159
eb9a9b9a0870_image,None 1 0 0 1 1
1177e1bef164_image,opacity 0.287795 54 114 98 180
a522fd19d916_image,opacity 0.445015 166 218 86 186
714c44f479b0_image,None 1 0 0 1 1
beb51e05dbc5_image,None 1 0 0 1 1
3a20fde98cdc_image,None 1 0 0 1 1
c5d9340b49ee_image,None 1 0 0 1 1
e3179eb2ff82_image,None 1 0 0 1 1
8be9683b4acc_image,None 1 0 0 1 1
849c83d5af12_image,opacity 0.341812 168 224 60 186 opacity 0.447705 31 87 69 177
f97bf716e365_image,None 1 0 0 1 1
ba2d7962e757_image,None 1 0 0 1 1
961a766a5511_image,opacity 0.450102 165 223 116 180
5af15b21333b_image,opacity 0.580503 139 197 44 204 opacity 0.660341 31 103 43 191
80e7a6e59bde_image,None 1 0 0 1 1
b7eb3f96fb93_image,opacity 0.52689 35 103 69 181 opacity 0.664728 143 205 82 198
9eb5cb68d2ca_image,None 1 0 0 1 1
a8217ee2d2a0_image,opacity 0.609138 31 83 60 174
d866b471de43_image,None 1 0 0 1 1
ffb8115a304c_image,None 1 0 0 1 1
2aaab6a41f1a_image,None 1 0 0 1 1
1450e124011a_image,None 1 0 0 1 1
875cf71bd22a_image,None 1 0 0 1 1
aee5e3feb1a2_image,opacity 0.359085 52 108 40 172 opacity 0.510828 156 208 36 186
5fbec0b4056c_image,opacity 0.638824 158 244 96 240 opacity 0.684502 38 118 97 241
c47fc2a4baaf_image,None 1 0 0 1 1
eeeb7da65659_image,None 1 0 0 1 1
78adeeaa636c_image,opacity 0.432794 153 201 92 172 opacity 0.518488 59 111 56 156
e85bb0bcca1f_image,None 1 0 0 1 1
b854ff324fbb_image,opacity 0.435203 2 104 66 200 opacity 0.668487 159 253 56 184
9998acc77910_image,opacity 0.296923 159 211 66 166 opacity 0.422518 42 104 60 162
4acbd31eafd5_image,None 1 0 0 1 1
82c5cf0b451c_image,opacity 0.287991 45 105 113 165 opacity 0.334954 159 211 96 178
95a69615f643_image,opacity 0.281978 23 67 71 199
ccb685c76038_image,None 1 0 0 1 1
05e168e0d636_image,None 1 0 0 1 1
b93887026df6_image,opacity 0.285901 154 222 134 202
f78ec02bc8d8_image,opacity 0.298887 59 115 67 163 opacity 0.377919 159 207 106 182
3f1439b85338_image,opacity 0.485667 68 116 54 180
f1774ffc9c4b_image,opacity 0.406521 63 133 80 188 opacity 0.496972 166 250 82 214
858667501a67_image,opacity 0.477528 48 120 65 187
26f55b8a7a94_image,opacity 0.417621 30 98 75 215
df937504e546_image,None 1 0 0 1 1
129846482a83_image,None 1 0 0 1 1
fcc09eaa77c4_image,None 1 0 0 1 1
b150add375b5_image,opacity 0.407456 58 112 91 191 opacity 0.470422 180 228 102 196
781a92a65529_image,opacity 0.313645 137 201 169 241
11abda231924_image,None 1 0 0 1 1
7750ed45ff97_image,opacity 0.359134 104 196 2 140 opacity 0.625273 0 74 1 109
11f8052288c7_image,opacity 0.482188 39 127 1 121 opacity 0.644748 157 253 0 96
e61e420938b6_image,opacity 0.351002 157 211 48 192
e8d801584d89_image,None 1 0 0 1 1
5fbc3a77d700_image,opacity 0.310458 182 234 100 198 opacity 0.575237 25 81 74 182
789ef0f544c3_image,opacity 0.515176 29 109 78 212 opacity 0.597002 157 237 80 224
d0d3d60f8914_image,None 1 0 0 1 1
949ea39318e2_image,opacity 0.455357 159 223 69 177
0cccb1eca1fc_image,None 1 0 0 1 1
223115cbf342_image,opacity 0.35829 173 241 100 174
b7ec5afe1084_image,opacity 0.573783 147 207 108 232 opacity 0.588498 29 109 100 228
1fb5ca50e8d3_image,opacity 0.385054 153 209 66 158
51cbf7662bee_image,opacity 0.307674 30 90 7 93
dea7079400f8_image,None 1 0 0 1 1
8e4fd6198ae9_image,None 1 0 0 1 1
beeb1db9f010_image,None 1 0 0 1 1
271bea69026c_image,opacity 0.399888 42 104 75 193
53316aefede6_image,opacity 0.516626 20 100 100 204
ee27ca277911_image,opacity 0.310308 3 79 30 166
47a976677416_image,None 1 0 0 1 1
a8de2b8aae69_image,opacity 0.359631 39 105 41 177 opacity 0.408178 171 231 54 188
b96333d66af7_image,opacity 0.573675 142 194 49 163 opacity 0.64446 39 103 31 165
7e3c0527ddb7_image,opacity 0.360856 173 233 129 229 opacity 0.452301 27 109 58 198
91558723b2cd_image,opacity 0.340824 138 198 88 208 opacity 0.507562 8 88 53 189
3b1dcd10fa0e_image,None 1 0 0 1 1
56414fb3f177_image,opacity 0.51501 32 80 62 204 opacity 0.558238 170 222 60 190
fe508ca896d7_image,None 1 0 0 1 1
2d6a4bc38540_image,opacity 0.425616 129 187 52 170
50c0e40fd929_image,opacity 0.444884 175 231 84 208
b5a75701a05e_image,None 1 0 0 1 1
ebcc9bedfa05_image,None 1 0 0 1 1
e8b4525ec41f_image,None 1 0 0 1 1
01c3512eebc3_image,None 1 0 0 1 1
7529a3e9745b_image,opacity 0.31635 53 125 59 207 opacity 0.526027 162 256 38 194
b37ae8cb220b_image,None 1 0 0 1 1
db4cecc08fe8_image,None 1 0 0 1 1
43e370c03cd2_image,None 1 0 0 1 1
e1635f289c95_image,None 1 0 0 1 1
82af7f7081d7_image,opacity 0.546051 0 92 7 147 opacity 0.639178 97 197 1 121
633e41912f6e_image,opacity 0.514614 -1 75 64 202 opacity 0.632164 103 183 60 196
0e97cc4123aa_image,opacity 0.503208 172 230 68 188
36e1a455aafc_image,None 1 0 0 1 1
b4eed879daf4_image,None 1 0 0 1 1
1fba9eccac03_image,opacity 0.325533 40 96 33 125
b1115b344129_image,opacity 0.337832 137 195 54 114
7d68127f8818_image,None 1 0 0 1 1
5765f6b4707d_image,opacity 0.302123 124 172 63 151
17ac0fe388c7_image,None 1 0 0 1 1
d997c6c6a9b4_image,opacity 0.551414 28 92 48 172 opacity 0.619113 144 208 39 163
d4ffcfab21d4_image,None 1 0 0 1 1
82eed39dfb6b_image,None 1 0 0 1 1
1ba8aa7acf76_image,None 1 0 0 1 1
4c0658c05a64_image,None 1 0 0 1 1
eb5787b68520_image,None 1 0 0 1 1
6ce67071aa0e_image,opacity 0.374746 67 125 46 170
363fbf00088c_image,None 1 0 0 1 1
23d8a619f39c_image,opacity 0.288058 60 132 97 253 opacity 0.54183 174 250 111 253
dc6834a1efa6_image,None 1 0 0 1 1
afc074e1bdb7_image,opacity 0.302962 165 221 61 217 opacity 0.521315 35 111 57 179
409322480a9b_image,opacity 0.438151 45 87 72 150
d1e5994d79de_image,opacity 0.284596 34 80 72 172
84680684e418_image,None 1 0 0 1 1
951211f8e1bb_image,None 1 0 0 1 1
8f925628e0f2_image,opacity 0.506916 39 103 59 179 opacity 0.551562 146 198 83 207
dfc5c09a50bc_image,opacity 0.437425 63 111 71 149
a492841dcf21_image,opacity 0.43993 143 209 42 182 opacity 0.458231 27 107 36 164
208c4c416450_image,None 1 0 0 1 1
9850b5470fd6_image,opacity 0.484093 28 110 69 209
62eca401e981_image,opacity 0.325348 186 238 107 175
db9da8404d58_image,None 1 0 0 1 1
b1e35564ea39_image,None 1 0 0 1 1
b6e6bb7b58d1_image,None 1 0 0 1 1
36a2991ffe0e_image,opacity 0.308669 40 106 137 209
9b934d581e5d_image,None 1 0 0 1 1
0d617e83e527_image,None 1 0 0 1 1
629ef3bd6a66_image,opacity 0.329732 26 102 78 218 opacity 0.547402 149 211 105 229
8cf062b99674_image,None 1 0 0 1 1
04f4a36c08a2_image,opacity 0.358567 7 73 22 116 opacity 0.468767 132 196 21 133
50201fe80b1b_image,opacity 0.351989 158 210 96 186 opacity 0.564469 39 105 62 178
59c4bfbf8b5b_image,None 1 0 0 1 1
e226747a2d76_image,opacity 0.476057 23 87 63 199 opacity 0.533097 133 205 36 194
95ed767a23e3_image,None 1 0 0 1 1
bfaffdf740f5_image,None 1 0 0 1 1
04720d7e4e42_image,opacity 0.287182 62 122 97 189
11b8e5179115_image,opacity 0.28573 154 214 129 193
904141862f44_image,None 1 0 0 1 1
8f1894f52499_image,opacity 0.378734 167 221 121 229
03a778f5a68b_image,opacity 0.614374 158 218 67 211 opacity 0.628131 31 115 63 207
7571cda597cb_image,opacity 0.356721 66 146 162 226
91e5e9e37c58_image,opacity 0.508352 0 80 56 168 opacity 0.545031 122 182 61 185
e150241b43a4_image,opacity 0.520745 3 83 45 161 opacity 0.605865 121 183 49 169
de1b8900bf84_image,opacity 0.413998 43 99 62 170 opacity 0.59979 151 215 101 199
816bb3680b2a_image,None 1 0 0 1 1
62fb484194fb_image,opacity 0.293132 127 175 100 204
35d6801ca10a_image,opacity 0.382274 29 85 109 175 opacity 0.514618 139 203 105 173
5e8d99b9ab38_image,opacity 0.521215 31 103 49 201 opacity 0.612876 145 209 49 197
5bab1b955f2a_image,None 1 0 0 1 1
9b3eae4045c8_image,opacity 0.316969 60 120 96 196
144c27a2554a_image,None 1 0 0 1 1
13c550e44b7c_image,None 1 0 0 1 1
d0e4465e3731_image,opacity 0.539443 168 248 67 171 opacity 0.59249 51 111 58 174
b57e134bbaef_image,opacity 0.296351 54 106 69 181 opacity 0.463722 161 225 68 170
d42cadfaac8f_image,None 1 0 0 1 1
99978f106a39_image,None 1 0 0 1 1
50dfcc0c2826_image,opacity 0.437526 129 199 69 209 opacity 0.486552 19 95 78 192
af6abf791a39_image,None 1 0 0 1 1
3e4b4468eca3_image,opacity 0.485125 159 219 85 221 opacity 0.581307 48 112 95 247
fb073252b364_image,None 1 0 0 1 1
2083ae742497_image,opacity 0.36646 147 217 46 194 opacity 0.453149 25 105 45 177
6cc1a9065abb_image,opacity 0.365894 147 217 46 194 opacity 0.452816 25 105 45 177
252552312a42_image,None 1 0 0 1 1
7553bffc2232_image,None 1 0 0 1 1
8887b33c4bb3_image,None 1 0 0 1 1
59a0fc64bd20_image,opacity 0.284825 137 229 57 171 opacity 0.360186 3 69 58 182
f02e557a36dc_image,opacity 0.284825 137 229 57 171 opacity 0.360186 3 69 58 182
127c2b32dd5d_image,opacity 0.296831 0 74 74 188
97de5453f769_image,opacity 0.296249 40 126 58 176
318a1ecb6fc7_image,opacity 0.296249 40 126 58 176
8c6d6a69d5dd_image,None 1 0 0 1 1
3b4ab121cbd4_image,None 1 0 0 1 1
2e131066f152_image,None 1 0 0 1 1
bcefb8e50563_image,None 1 0 0 1 1
363dd47c8b10_image,opacity 0.311829 41 85 96 184
96780bd04f67_image,opacity 0.434561 51 111 123 191
731910f93559_image,None 1 0 0 1 1
eaa1d528538f_image,None 1 0 0 1 1
760acda93997_image,None 1 0 0 1 1
d0e829b775fe_image,None 1 0 0 1 1
440de3bfe05a_image,None 1 0 0 1 1
55433651ec05_image,None 1 0 0 1 1
a72e1270f4cd_image,opacity 0.369747 7 95 51 207 opacity 0.404597 140 202 77 201
5699fc89704f_image,opacity 0.350638 139 195 49 141 opacity 0.526757 46 98 63 165
aa55fc6c616c_image,None 1 0 0 1 1
58bc298e0f9f_image,opacity 0.349158 41 109 103 199 opacity 0.426612 154 218 125 217
5adf919e765a_image,None 1 0 0 1 1
bb425c8a4aae_image,None 1 0 0 1 1
5cb984539107_image,opacity 0.335154 14 90 124 220
079be2a10982_image,None 1 0 0 1 1
14db2ff155f6_image,None 1 0 0 1 1
56a3fb6663bf_image,opacity 0.309832 148 198 52 182
3a73591225ea_image,None 1 0 0 1 1
a2460d88071d_image,opacity 0.341155 157 225 77 157
c3ded8550e66_image,None 1 0 0 1 1
cdcfbf2ca285_image,None 1 0 0 1 1
303840f71bb5_image,opacity 0.61555 155 221 63 171 opacity 0.6683 37 113 58 190
11349660bb1a_image,opacity 0.466288 51 99 56 144 opacity 0.481171 155 207 58 152
94ddca484ebf_image,None 1 0 0 1 1
21d1f013892c_image,None 1 0 0 1 1
3288907ddc51_image,None 1 0 0 1 1
6e798a5025b7_image,None 1 0 0 1 1
1d674718db40_image,opacity 0.405375 56 112 35 167 opacity 0.483483 156 212 22 178
29c992128fbd_image,None 1 0 0 1 1
b4d04b517ef8_image,None 1 0 0 1 1
02eceb0fc405_image,None 1 0 0 1 1
c43e88a44468_image,None 1 0 0 1 1
1fcfc9845e81_image,None 1 0 0 1 1
718676f11609_image,None 1 0 0 1 1
35dd17ea3e6a_image,opacity 0.355559 37 91 69 185
e831bebfa015_image,opacity 0.35029 39 95 17 153
49c1b06028a1_image,opacity 0.531031 55 107 87 197
a2c0d74fda88_image,None 1 0 0 1 1
9bc2041c6ac0_image,opacity 0.370375 61 107 112 186
8757ce9b582c_image,None 1 0 0 1 1
0c81b34981b3_image,None 1 0 0 1 1
f8ed7c6111f1_image,opacity 0.629123 40 104 30 182 opacity 0.681688 140 206 30 174
ff5448be90d5_image,None 1 0 0 1 1
1047ca1a704c_image,None 1 0 0 1 1
cdc6e15cc1a6_image,opacity 0.511142 124 176 59 177 opacity 0.63449 22 84 52 158
65a5d46ef968_image,opacity 0.284676 167 215 104 196
8b3366ec66d4_image,None 1 0 0 1 1
a84c79aac3c0_image,opacity 0.439986 135 227 71 199
daf0bd7ba1b4_image,opacity 0.559419 136 224 78 200
620191dbdfa4_image,None 1 0 0 1 1
81eb1b621a83_image,None 1 0 0 1 1
d4f61c144f48_image,opacity 0.314557 29 93 46 184 opacity 0.528672 160 216 43 175
f3f3ea70d5a7_image,None 1 0 0 1 1
a88f38a36bb2_image,None 1 0 0 1 1
d40d1b28e300_image,opacity 0.526801 59 135 36 204 opacity 0.633106 160 252 40 192
fcdb115a918f_image,opacity 0.438006 66 126 61 213 opacity 0.579991 168 230 68 232
1c47d2fcee72_image,None 1 0 0 1 1
8b901c5f333a_image,opacity 0.298961 55 123 83 151 opacity 0.36358 172 228 104 152
c0b95ee9cc9f_image,None 1 0 0 1 1
93ddd4e6eb4c_image,None 1 0 0 1 1
66ca7bb52ab9_image,None 1 0 0 1 1
458417ad6155_image,opacity 0.283345 34 86 101 185
5b27050cfb90_image,None 1 0 0 1 1
f1b66ff8f4eb_image,opacity 0.367742 155 207 43 167 opacity 0.595304 53 109 22 154
5db9599cf9f7_image,opacity 0.438484 160 220 83 211 opacity 0.440925 37 109 53 217
45bc12129df2_image,None 1 0 0 1 1
acd735d74d02_image,None 1 0 0 1 1
c00bae0e7714_image,None 1 0 0 1 1
37811132dc96_image,None 1 0 0 1 1
d88e65063aef_image,None 1 0 0 1 1
7731db2caed2_image,opacity 0.574067 165 229 56 192 opacity 0.682093 38 128 46 186
5fa584b8b4ad_image,opacity 0.364851 58 106 61 145
6758e5ed7742_image,opacity 0.28482 41 113 132 196 opacity 0.334309 175 239 133 205
1b0d0622ccea_image,None 1 0 0 1 1
c3d0378f8856_image,opacity 0.431745 152 208 103 165
d750be1736db_image,opacity 0.529392 14 94 66 206 opacity 0.597455 133 203 56 222
bf5c7ef3b72a_image,opacity 0.502703 175 229 73 185 opacity 0.525875 37 85 82 182
75a172588bc8_image,opacity 0.500028 32 82 66 174 opacity 0.549925 168 224 45 175
b081d24bd60f_image,opacity 0.301519 14 80 74 150 opacity 0.460445 136 198 44 148
64c64a317c96_image,opacity 0.393974 42 98 84 212 opacity 0.433239 157 217 67 217
a6fae2436886_image,None 1 0 0 1 1
71a354af0223_image,None 1 0 0 1 1
27a4a0d2b7ec_image,opacity 0.338143 41 89 56 172 opacity 0.375752 167 219 92 148
e2c038335c9a_image,opacity 0.284749 151 211 115 175
7e11aff74953_image,None 1 0 0 1 1
ecebf3314fbd_image,None 1 0 0 1 1
164dd63ae571_image,opacity 0.308616 54 114 45 171
adb60ba82edf_image,opacity 0.580323 141 201 13 147 opacity 0.644058 34 102 12 150
8180af690aef_image,None 1 0 0 1 1
deb2db0a0efb_image,None 1 0 0 1 1
89128c0fa9f4_image,None 1 0 0 1 1
2f8c1d151802_image,None 1 0 0 1 1
acab76326ba1_image,opacity 0.476909 45 93 82 222 opacity 0.510417 175 239 85 225
abee6a2d0bf5_image,opacity 0.406372 181 233 74 174
0a0c1312be06_image,None 1 0 0 1 1
173c3e37e1f3_image,None 1 0 0 1 1
e7b2a9418b98_image,opacity 0.568833 14 102 45 189 opacity 0.636356 137 209 45 193
be6bf80c7071_image,None 1 0 0 1 1
faf630376fa9_image,None 1 0 0 1 1
36ba388a18df_image,opacity 0.449442 24 72 110 210
54066d711ef9_image,None 1 0 0 1 1
7ea12126a6e3_image,None 1 0 0 1 1
e20cf732e8eb_image,None 1 0 0 1 1
670989e56968_image,opacity 0.315768 53 113 79 161
b4b6e82b3dc9_image,opacity 0.347318 53 117 114 178
bca76008a759_image,opacity 0.402918 145 209 72 208
404d53460643_image,opacity 0.288446 169 227 51 179
37c6c0aee54b_image,opacity 0.569488 29 115 45 179 opacity 0.67966 161 241 44 192
11b9ab3af3db_image,None 1 0 0 1 1
b362aec0b6c5_image,None 1 0 0 1 1
d5a056bedf60_image,opacity 0.326617 13 85 121 193
fe501aa91e43_image,opacity 0.372699 146 200 61 185 opacity 0.451233 41 97 50 174
eaec01e48714_image,opacity 0.594686 27 95 65 215 opacity 0.629455 132 216 56 200
fc332a1e51c9_image,None 1 0 0 1 1
1c69c1af9a67_image,None 1 0 0 1 1
25b281d5a9f3_image,opacity 0.580609 157 207 63 199 opacity 0.623406 62 122 59 187
b74f81d65e79_image,None 1 0 0 1 1
05ab2c886dc1_image,None 1 0 0 1 1
4f83ec489a15_image,opacity 0.365612 164 252 4 108 opacity 0.512297 54 118 37 145
f3239829a8dc_image,opacity 0.40181 35 95 61 205 opacity 0.445526 129 189 63 213
caa431721668_image,None 1 0 0 1 1
7509f6912049_image,None 1 0 0 1 1
c5a76564ef83_image,None 1 0 0 1 1
7a608ecc4868_image,opacity 0.495647 37 101 84 188 opacity 0.627248 146 204 75 163
c22134ea64bc_image,None 1 0 0 1 1
7bba27555fe4_image,None 1 0 0 1 1
bbb202078733_image,opacity 0.449708 40 104 47 179 opacity 0.470095 146 206 65 177
53b4af5b74d7_image,None 1 0 0 1 1
a356db474991_image,None 1 0 0 1 1
14eb73431579_image,None 1 0 0 1 1
0dc27df9b1bf_image,None 1 0 0 1 1
793167fdec10_image,None 1 0 0 1 1
beb80bb88a6e_image,opacity 0.297955 52 92 68 172
26da12104c4d_image,None 1 0 0 1 1
6707b082f3c9_image,opacity 0.413555 163 245 27 181 opacity 0.440414 34 126 25 173
7bbafcdea0cc_image,None 1 0 0 1 1
51d9dda5e493_image,None 1 0 0 1 1
cca081560d98_image,opacity 0.329013 7 75 62 164 opacity 0.453974 128 196 96 194
4ccb8a39c77a_image,None 1 0 0 1 1
d35dc6f25249_image,opacity 0.347167 66 118 43 105
cd6acf7eecab_image,opacity 0.311908 175 231 68 184 opacity 0.328507 42 102 90 212
17d708d24ae8_image,None 1 0 0 1 1
71dc3bd13f18_image,opacity 0.462807 162 230 99 191 opacity 0.584155 44 116 73 183
6be23f5c40f3_image,None 1 0 0 1 1
96948f97f356_image,opacity 0.282454 176 222 67 169
fa0c84ee4577_image,opacity 0.443134 142 190 55 183 opacity 0.647342 45 105 42 166
f91fbbfe4682_image,opacity 0.423424 173 229 78 190 opacity 0.485826 73 129 64 172
f2d98750a22c_image,opacity 0.409178 63 107 65 181
ab0a95dee3b0_image,None 1 0 0 1 1
89ee5be93326_image,opacity 0.397291 167 217 89 197
d7aaf0a52df3_image,opacity 0.397291 167 217 89 197
1aff7e4ed470_image,opacity 0.476664 34 86 52 148
895701fa488e_image,None 1 0 0 1 1
abd45a0023eb_image,opacity 0.292212 159 205 56 176
e252ffbf487e_image,None 1 0 0 1 1
fce43e07637a_image,opacity 0.436134 69 123 62 146 opacity 0.437539 167 221 80 158
56b010725833_image,None 1 0 0 1 1
fe2c1dc506e7_image,None 1 0 0 1 1
f2be75ff15ba_image,None 1 0 0 1 1
fe0cf147b965_image,None 1 0 0 1 1
3b5f48db44cc_image,opacity 0.285359 54 94 42 152
022146012034_image,None 1 0 0 1 1
dc58aecf62e3_image,opacity 0.323924 129 213 51 179 opacity 0.515053 17 101 45 203
413344e1032c_image,opacity 0.296575 60 116 100 188
64ea82b1343f_image,opacity 0.539207 19 67 61 175
6f423b9b0633_image,None 1 0 0 1 1
1cbd87cb6bf9_image,None 1 0 0 1 1
2de24d1c32bb_image,opacity 0.341247 41 101 115 163
c3afb9da2ebc_image,None 1 0 0 1 1
04a083828fec_image,None 1 0 0 1 1
897e2799654c_image,None 1 0 0 1 1
cc495995f8f5_image,None 1 0 0 1 1
08b12b305a50_image,opacity 0.40178 30 94 71 223
ac238e67fdc5_image,None 1 0 0 1 1
1c5aa8c405c7_image,None 1 0 0 1 1
5a2b8c736aa3_image,None 1 0 0 1 1
8ea54e20fd42_image,opacity 0.297582 161 213 96 180
a43200bd5ceb_image,None 1 0 0 1 1
60197b6237e4_image,None 1 0 0 1 1
d393233d1b40_image,None 1 0 0 1 1
88debce8e1c3_image,None 1 0 0 1 1
dc5141a2316a_image,opacity 0.363752 161 209 70 174 opacity 0.387745 56 104 78 186
fd322d964dc7_image,opacity 0.478416 153 225 114 196
88e9dabee5fc_image,None 1 0 0 1 1
4d6c8e93e7c9_image,opacity 0.282681 181 225 49 189 opacity 0.290683 64 104 56 190
1f49246b305e_image,None 1 0 0 1 1
e95803934fc4_image,None 1 0 0 1 1
70bee63a6740_image,opacity 0.521577 44 116 69 221
27d3da7a8e3d_image,None 1 0 0 1 1
79e957055a36_image,opacity 0.306747 155 209 65 201
85b6452b47cc_image,opacity 0.398211 14 66 55 171
1ab561ec5f7e_image,opacity 0.478058 181 257 97 237
cede944b70a5_image,None 1 0 0 1 1
1dfed0992998_image,opacity 0.460414 146 198 85 189 opacity 0.480916 51 107 71 187
6d3f777f0a8c_image,None 1 0 0 1 1
04de800ea41a_image,opacity 0.284222 49 93 91 163
9c460da48071_image,None 1 0 0 1 1
26102c3f4f0f_image,opacity 0.382598 153 205 87 193 opacity 0.518666 37 105 56 174
e420d66bb7f8_image,opacity 0.31826 58 118 67 167
82da4f94ce03_image,None 1 0 0 1 1
9cd592696284_image,None 1 0 0 1 1
4d56b2fc668c_image,opacity 0.341921 33 97 61 171 opacity 0.543938 159 219 67 183
69ffab5539ec_image,opacity 0.377934 150 226 101 199
03fa9d7e94a8_image,None 1 0 0 1 1
07db3756ec28_image,opacity 0.407603 148 212 35 197 opacity 0.445029 26 92 36 204
9271b6df7062_image,None 1 0 0 1 1
955bf1edd532_image,None 1 0 0 1 1
f016fe5579d7_image,None 1 0 0 1 1
f07e4a04078a_image,None 1 0 0 1 1
516e4adba012_image,None 1 0 0 1 1
1137ee93363f_image,opacity 0.326802 176 246 125 217 opacity 0.575007 63 121 119 223
ad98af65ad2a_image,opacity 0.588538 40 112 88 184 opacity 0.663335 161 233 98 214
059eae61ad15_image,None 1 0 0 1 1
4c56fe85fa4f_image,None 1 0 0 1 1
7d27573ab0c5_image,None 1 0 0 1 1
33d456320f28_image,None 1 0 0 1 1
ba0cf340582d_image,opacity 0.403404 63 115 85 201 opacity 0.492787 149 211 98 192
65acc1fb99a3_image,None 1 0 0 1 1
896d667947af_image,None 1 0 0 1 1
6dd0cdbd225f_image,opacity 0.367564 38 98 49 169 opacity 0.541972 164 228 48 184
251f897c9e10_image,None 1 0 0 1 1
ebf5edbbed2d_image,opacity 0.461481 26 94 134 234
2955c5ce3f0f_image,None 1 0 0 1 1
a37a362df0ac_image,None 1 0 0 1 1
854f9f3e562f_image,None 1 0 0 1 1
6a5a6eaef699_image,opacity 0.322107 162 222 111 167
edd69153435a_image,None 1 0 0 1 1
da66593f5d16_image,opacity 0.345719 34 96 124 180
33e0c978d0da_image,opacity 0.311507 30 90 53 165
abe007e3fc08_image,None 1 0 0 1 1
b92e6f1b6c1b_image,opacity 0.448915 36 100 37 169 opacity 0.514129 146 204 55 183
948e9f366810_image,opacity 0.600151 68 136 78 222 opacity 0.615949 179 253 67 211
3cccbd897c39_image,None 1 0 0 1 1
3bb8935d30d9_image,None 1 0 0 1 1
37d6750b5282_image,opacity 0.289108 52 108 85 197 opacity 0.450337 153 209 87 187
67f192ad1a53_image,opacity 0.42714 147 199 50 164 opacity 0.579659 45 105 36 140
7f090fbd8e7c_image,None 1 0 0 1 1
898d24f3cda6_image,opacity 0.479425 17 105 35 171
a0f6e42396fc_image,opacity 0.37193 171 231 106 182 opacity 0.431259 67 127 61 179
422c0e145066_image,None 1 0 0 1 1
46719b856de1_image,None 1 0 0 1 1
31c07523a69a_image,opacity 0.427264 53 113 78 200 opacity 0.565582 168 238 58 194
f77d7d1aebab_image,None 1 0 0 1 1
ccc5b63ca96d_image,None 1 0 0 1 1
5e8ac1fe2b82_image,opacity 0.318599 166 232 126 242 opacity 0.427661 48 120 78 234
'''
import io
import pandas
import numpy
with io.StringIO(t2) as f:
    t3 = pandas.read_csv(f)
t3.PredictionString = t3.PredictionString.apply(lambda x: x.replace('Negative', 'negative').replace('None', 'none'))
for i in range(t3.shape[0]):
    t4 = t3.iloc[i].PredictionString
    if 'opacity' in t4:
        t5 = t4.split()
        assert len(t5) % 6 == 0, 'error'
        t6 = numpy.array(t5).reshape(-1, 6)
        assert numpy.all(t6[:, 0] == 'opacity'), 'error 2'
        #print(t6)
        assert numpy.all(t6[:, 2].astype(numpy.int32) < t6[:, 3].astype(numpy.int32)), 'error 3'
        assert numpy.all(t6[:, 4].astype(numpy.int32) < t6[:, 5].astype(numpy.int32)), 'error 4'
        t7 = numpy.stack([
            t6[:, 0],
            t6[:, 1],
            t6[:, 2], t6[:, 4],
            t6[:, 3], t6[:, 5]
        ]).T
        #print(t7)
        #print(i)
        t8 = ' '.join(t7.flatten().tolist())
        #print(t8)
        t3.iloc[i].PredictionString = t8
assert (submission_df['id'].values == t3['id'].values).all()

display(t3)


t3.to_csv('submission.csv', index=False)

#https://i.imgur.com/3ARpdM0.png
#incorrect format of labels, big letters, etc
    
t9 = t3.copy()
t9.PredictionString = submission_df.PredictionString
t9.iloc[2476].values[1] = 'opacity 0.318599 166 126 232 242 opacity 0.427661 48 78 120 234'
display(t9)
t9.to_csv('submission.csv', index=False)
    