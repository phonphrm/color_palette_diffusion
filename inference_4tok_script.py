import os

os.system("python inference_4tok.py --checkpoint_step=10000 --cfg=3 --seed=42")
os.system("python inference_4tok.py --checkpoint_step=10000 --cfg=5 --seed=42")
os.system("python inference_4tok.py --checkpoint_step=10000 --cfg=7.5 --seed=42")


os.system("python inference_4tok.py --checkpoint_step=20000 --cfg=3 --seed=42")
os.system("python inference_4tok.py --checkpoint_step=20000 --cfg=5 --seed=42")
os.system("python inference_4tok.py --checkpoint_step=20000 --cfg=7.5 --seed=42")


os.system("python inference_4tok.py --checkpoint_step=30000 --cfg=3 --seed=42")
os.system("python inference_4tok.py --checkpoint_step=30000 --cfg=5 --seed=42")
os.system("python inference_4tok.py --checkpoint_step=30000 --cfg=7.5 --seed=42")


os.system("python inference_4tok.py --checkpoint_step=40000 --cfg=3 --seed=42")
os.system("python inference_4tok.py --checkpoint_step=40000 --cfg=5 --seed=42")
os.system("python inference_4tok.py --checkpoint_step=40000 --cfg=7.5 --seed=42")

# os.system("python inference_4tok.py --checkpoint_step=50000 --cfg=3 --seed=42")
# os.system("python inference_4tok.py --checkpoint_step=50000 --cfg=5 --seed=42")
# os.system("python inference_4tok.py --checkpoint_step=50000 --cfg=7.5 --seed=42")