import os

data_dir = "../data/raw/fusion360gallery/reconstruction"
step_files = [f for f in os.listdir(data_dir) if f.endswith(".step")]

three_token = 0   # e.g. 20203_7e31e92a_0000.step  -> likely the actual target/final design
four_token = 0    # e.g. 20203_7e31e92a_0000_0001.step -> likely an intermediate build step

for f in step_files:
    parts = f.replace(".step", "").split("_")
    if len(parts) == 3:
        three_token += 1
    elif len(parts) == 4:
        four_token += 1

print("3-token files (likely final target designs):", three_token)
print("4-token files (likely intermediate build steps):", four_token)