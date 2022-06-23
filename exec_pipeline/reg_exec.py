import argparse
from DL_pipeline import DeepLearningMatch
from Ori_pipeline import OriginalMatch

parser = argparse.ArgumentParser(description='Registration of CT images and 3DRA images using Traditional method or DeepLearning method.')
parser.add_argument('reg_method', type=str, choices=['tra', 'dl'], help='Choose which method to use')
parser.add_argument('--3dra_path', dest='ra_p', type=str, help='Path of 3DRA images')
parser.add_argument('--ct_path', dest='ct_p', type=str, help='Path of CT images')
parser.add_argument('--save_path', dest='sa_p', type=str, help='Path of saving output')

args = parser.parse_args()
if args.reg_method == 'tra':
    ori_matcher = OriginalMatch(args.ra_p)
    ori_matcher.match(args.ct_p, args.sava_p)

if args.reg_method == 'dl':
    DL_matcher = DeepLearningMatch(args.ra_p)
    DL_matcher.match_to_ori_file(args.ct_p, args.sava_p)