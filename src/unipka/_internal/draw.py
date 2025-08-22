from typing import Dict, List
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from IPython.display import SVG, display




def get_neutral_base_name(ensemble: Dict[int, List[str]]) -> str:
    q_list = sorted(ensemble.keys())
    min_q = -int(min(q_list))
    return "A" if min_q == 0 else f"H<sub>{min_q}</sub>A"


def calc_base_name(neutral_base_name: str, target_charge: int) -> str:
    if neutral_base_name.startswith("H"):
        if neutral_base_name[1:].startswith("<sub>"):
            num_H_end = neutral_base_name.find("</sub>", 6)
            num_H = int(neutral_base_name[6:num_H_end])
        else:
            num_H_end = 1
            num_H = 1
    else:
        num_H_end = 0
        num_H = 0
    target_num_H = num_H + target_charge
    assert target_num_H >= 0
    target_base_name = ""
    if target_num_H == 1:
        target_base_name += "H"
    elif target_num_H > 1:
        target_base_name += f"H<sub>{target_num_H}</sub>"
    target_base_name += "A"
    if target_charge < -1:
        target_base_name += f"<sup>{-target_charge}-</sup>"
    elif target_charge == -1:
        target_base_name += "<sup>-</sup>"
    elif target_charge == 1:
        target_base_name += "<sup>+</sup>"
    elif target_charge > 1:
        target_base_name += f"<sup>{target_charge}+</sup>"
    return target_base_name


def draw_macrostate(macrostate: List[str], base_name: str):
    macrostate_mols = list(map(Chem.MolFromSmiles, macrostate))
    macrostate_size = len(macrostate_mols)
    legends = [f"{i + 1}-{base_name}" for i in range(macrostate_size)]
    img = MolsToGridImage(macrostate_mols, legends=legends, useSVG=True)
    display(SVG(img.data))
    return img


def draw_ensemble(ensemble: Dict[int, List[str]]) -> None:
    q_list = sorted(ensemble.keys())
    neutral_base_name = get_neutral_base_name(ensemble)
    for q in q_list:
        draw_macrostate(ensemble[q], calc_base_name(neutral_base_name, q))

