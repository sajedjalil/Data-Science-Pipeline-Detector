#!/usr/bin/env python
# -*- coding: utf-8 -*-
# related to forum "Evaluation Score Map@10 ???"

import pandas as pd

sample = pd.read_csv('../input/sample_submission.csv')
sample.PURCHASED_COUPONS = "2fcca928b8b3e9ead0f2cecffeea50c1 0fd38be174187a3de72015ce9d5ca3a2"

sample.to_csv("lucky2coupons.csv", index=False)
