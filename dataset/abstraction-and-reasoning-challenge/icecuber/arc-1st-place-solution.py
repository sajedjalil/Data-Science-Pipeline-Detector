
print("Running")
from subprocess import Popen, PIPE, STDOUT
from glob import glob

if open("../input/arc-solution-source-files/version.txt").read().strip() == "671838222":
  print("Dataset has correct version")
else:
  print("Dataset version not matching!")
  assert(0)

def mySystem(cmd):
    print(cmd)
    process = Popen(cmd, stdout=PIPE, stderr=STDOUT, shell=True)
    for line in iter(process.stdout.readline, b''):
        print(line.decode("utf-8"), end='')
    assert(process.wait() == 0)
dummy_run = False


for fn in glob("/kaggle/input/abstraction-and-reasoning-challenge/test/*.json"):
  if "136b0064" in fn:
    print("Making dummy submission")
    f = open("submission.csv", "w")
    f.write("output_id,output\n")
    f.close()
    dummy_run = True


if not dummy_run:
  mySystem("cp -r ../input/arc-solution-source-files ./absres-c-files")
  mySystem("cd absres-c-files; make -j")
  mySystem("cd absres-c-files; python3 safe_run.py")
  mySystem("cp absres-c-files/submission_part.csv submission.csv")
  mySystem("tar -czf store.tar.gz absres-c-files/store")
  mySystem("rm -r absres-c-files")
print("Done")
