import os
import re
import yaml
from pathlib import Path
from copy import deepcopy

def generate_param_variants_flat(base_config_dir: str, param_dict: dict):
    base_path = Path(base_config_dir)
    last_dir = base_path.name
    capitalized = last_dir.capitalize()
    base_file = base_path / f'config_{last_dir}_1.yml'

    if not base_file.exists():
        raise FileNotFoundError(f"Base config file {base_file} not found")

    with open(base_file, 'r') as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    base_expname = base_config['exp_name']
    match = re.search(rf"{capitalized}_param_(\d+)", base_expname)
    if not match:
        raise ValueError("exp_name format invalid")
    base_idx = int(match.group(1))
    file_count = 0

    for key, values in param_dict.items():
        for v in values:
            new_config = deepcopy(base_config)
            new_config[key] = v

            file_count += 1
            new_idx = base_idx + file_count

            # exp_name の更新
            new_expname = re.sub(
                rf"{capitalized}_param_\d+",
                f"{capitalized}_param_{new_idx}",
                base_expname
            )
            new_config['exp_name'] = new_expname

            # 出力パス例: NAF_logs/pra_ch_emb/config_pra_ch_emb_param_7.yml
            output_file = base_path / f'config_{last_dir}_param_{new_idx}.yml'
            with open(output_file, 'w') as f:
                yaml.dump(new_config, f, sort_keys=False)

            print(f"✅ Generated: {output_file}")

    print(f"✔️ Total YAML files generated: {file_count}")

if __name__ == "__main__":
    param_dict = {
        "lr_init": [0.005, 0.0005, 0.00005,],
    }

    generate_param_variants_flat("config_files/pra", param_dict)
    generate_param_variants_flat("config_files/pra_ch_emb", param_dict)
    generate_param_variants_flat("config_files/real_env", param_dict)
    generate_param_variants_flat("config_files/real_env_ch_emb", param_dict)
    generate_param_variants_flat("config_files/real_exp", param_dict)
    generate_param_variants_flat("config_files/real_exp_ch_emb", param_dict)
