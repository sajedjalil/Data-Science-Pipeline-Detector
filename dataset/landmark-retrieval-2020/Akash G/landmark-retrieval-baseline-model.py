from zipfile import ZipFile

with ZipFile('submission.zip','w') as zip:           
  zip.write('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model/saved_model.pb', arcname='saved_model.pb') 
  zip.write('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model/variables/variables.data-00000-of-00001', arcname='variables/variables.data-00000-of-00001') 
  zip.write('../input/baseline-landmark-retrieval-model/baseline_landmark_retrieval_model/variables/variables.index', arcname='variables/variables.index') 