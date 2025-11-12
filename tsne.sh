# # python tsne_plot.py \
# #   --repo-roots "src:src_crossfreq_vit" \
# #   --data-root "/home/UNT/ak2102/DeepLearning_project/vit_vs_resnet/data/imagenet_subset" \
# #   --split val \
# #   --n-classes 20 \
# #   --per-class 600 \
# #   --batch-size 64 \
# #   --workers 4 \
# #   --models \
# #   "name=resnet,module=src.models,class=build_resnet152,args='{\"num_classes\":100}',ckpt=checkpoints/resnet152_imagenet_pretrained_imagenet_subset_1.0_20251111-203401.pth,feature_layer=fc" \
# #   "name=vit_b16,module=src.models,class=build_vit_b16,args='{\"num_classes\":100}',ckpt=checkpoints/vit_b16_imagenet_pretrained_imagenet_subset_1.0_20251111-222057.pth,feature_layer=heads.head" \
# #   "name=freqaware,module=src_crossfreq_vit.model_crossfreq_vit,class=CrossFreqViT,args='{\"num_classes\":100}',ckpt=checkpoints/best_crossfreq_vit.pth,feature_layer=head.fc" \
# #   --out "tsne_joint_10classes.png"

# python tsne_plot.py \
#   --repo-roots "src:src_crossfreq_vit" \
#   --data-root "/home/UNT/ak2102/DeepLearning_project/vit_vs_resnet/data/imagenet_subset" \
#   --split val \
#   --n-classes 30 \
#   --per-class 60 \
#   --batch-size 64 \
#   --workers 4 \
#   --models \
#   "name=resnet,module=src.models,class=build_resnet152,args='{\"num_classes\":100}',ckpt=checkpoints/resnet152_imagenet_pretrained_imagenet_subset_1.0_20251111-203401.pth,feature_layer=fc" \
#   "name=vit_b16,module=src.models,class=build_vit_b16,args='{\"num_classes\":100}',ckpt=checkpoints/vit_b16_imagenet_pretrained_imagenet_subset_1.0_20251111-222057.pth,feature_layer=heads.head" \
#   "name=freqaware,module=src_crossfreq_vit.model_crossfreq_vit,class=build_vit_crossfreq,args='{\"num_classes\":100}',ckpt=checkpoints/best_crossfreq_vit.pth,feature_layer=classifier" \
#   --out "tsne_joint_10classes.png"

python tsne_plot.py \
  --repo-roots "src:src_crossfreq_vit" \
  --data-root "/home/UNT/ak2102/DeepLearning_project/vit_vs_resnet/data/imagenet_subset" \
  --split val \
  --n-classes 30 \
  --per-class 60 \
  --batch-size 64 \
  --workers 4 \
  --models \
  "name=resnet,module=src.models,class=build_resnet152,args='{\"num_classes\":100}',ckpt=checkpoints/resnet152_imagenet_pretrained_imagenet_subset_1.0_20251111-203401.pth,feature_layer=fc" \
  "name=vit_b16,module=src.models,class=build_vit_b16,args='{\"num_classes\":100}',ckpt=checkpoints/vit_b16_imagenet_pretrained_imagenet_subset_1.0_20251111-222057.pth,feature_layer=heads.head" \
  "name=freqaware,module=src_crossfreq_vit.model_crossfreq_vit,class=build_vit_crossfreq,args='{\"num_classes\":100}',ckpt=checkpoints/best_crossfreq_vit.pth,feature_layer=classifier" \
  --out "tsne_joint_10classes.png"
