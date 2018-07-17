import shutil
import os

#cp -r /tmp/finetune-transformer-lm/model ../../models/openai-transformer-lm
#rm -rf /tmp/finetune-transformer-lm/

if __name__ == "__main__":
  os.system("git clone https://github.com/openai/finetune-transformer-lm /tmp/finetune-transformer-lm/")
  p = os.path.dirname(os.path.realpath(__file__))
  shutil.copytree("/tmp/finetune-transformer-lm/model", os.path.join(p, "../../models/openai-transformer-lm"))
  shutil.rmtree("/tmp/finetune-transformer-lm")