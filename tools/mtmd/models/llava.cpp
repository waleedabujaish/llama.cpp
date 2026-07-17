#include "models.h"

// this graph is used by llava, granite and glm
// due to having embedding_stack (used by granite), we cannot reuse build_vit
ggml_cgraph * clip_graph_llava::build() {
    const int batch_size = 1;
    const int n_pos = n_patches + (model.class_embedding ? 1 : 0);

    GGML_ASSERT(n_patches_x == n_patches_y && "only square images supported");

    // Calculate the deepest feature layer based on hparams and projector type
    int max_feature_layer = n_layer;
    {
        int il_last = hparams.n_layer;
        int deepest_feature_layer = -1;
        // If we set explicit vision feature layers, only go up to the deepest one
        // NOTE: only used by granite-vision models for now
        for (const auto & feature_layer : hparams.feature_layers) {
            if (feature_layer > deepest_feature_layer) {
                deepest_feature_layer = feature_layer;
            }
        }
        max_feature_layer = deepest_feature_layer < 0 ? il_last : deepest_feature_layer;
    }

    // visual token pruning (FasterVLM-style CLS-attention scoring), CLIP-family only.
    // visual_prune_method gates on top of visual_keep so future non-"cls" methods
    // don't silently activate this branch once they're added.
    const bool prune_visual_tokens =
        hparams.visual_keep < 1.0f &&
        hparams.visual_prune_method == "cls" &&
        model.class_embedding != nullptr &&
        proj_type == PROJECTOR_TYPE_MLP;

    ggml_tensor * inp = build_inp();

    // concat class_embeddings and patch_embeddings
    if (model.class_embedding) {
        inp = ggml_concat(ctx0, model.class_embedding, inp, 1);
    }

    ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_pos);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    inp = ggml_add(ctx0, inp, ggml_get_rows(ctx0, model.position_embeddings, positions));

    ggml_tensor * inpL = inp;

    // pre-layernorm
    if (model.pre_ln_w) {
        inpL = build_norm(inpL, model.pre_ln_w, model.pre_ln_b, NORM_TYPE_NORMAL, eps, -1);
        cb(inpL, "pre_ln", -1);
    }

    std::vector<ggml_tensor *> embedding_stack;

    // [n_patches] mean CLS-attention score per patch, set at the scoring layer
    // below when prune_visual_tokens is active; consumed by the top-K gather
    // that replaces the "patches" input further down.
    ggml_tensor * cls_scores = nullptr;

    // loop over layers
    for (int il = 0; il < max_feature_layer; il++) {
        auto & layer = model.layers[il];
        ggml_tensor * cur = inpL; // inpL = residual, cur = hidden_states

        // If this is an embedding feature layer, save the output.
        // NOTE: 0 index here refers to the input to the encoder.
        if (hparams.is_feature_layer(il)) {
            embedding_stack.push_back(cur);
        }

        // layernorm1
        cur = build_norm(cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "layer_inp_normed", il);

        // self-attention
        {
            ggml_tensor * Qcur = build_mm(layer.q_w, cur);
            if (layer.q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            }

            ggml_tensor * Kcur = build_mm(layer.k_w, cur);
            if (layer.k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.k_b);
            }

            ggml_tensor * Vcur = build_mm(layer.v_w, cur);
            if (layer.v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(layer.o_w, layer.o_b,
                Qcur, Kcur, Vcur, nullptr, kq_scale, il);
            cb(cur, "attn_out", il);

            // CLS-attention scoring branch: independent of the main attention
            // above, reuses the same Qcur/Kcur. Only built at the scoring layer
            // (the last built layer) and only when pruning is active.
            if (prune_visual_tokens && il == max_feature_layer - 1) {
                // CLS query only: row 0 of Qcur [d_head, n_head, n_pos]
                ggml_tensor * q_cls = ggml_view_3d(ctx0, Qcur, d_head, n_head, 1, Qcur->nb[1], Qcur->nb[2], 0);
                q_cls = ggml_permute(ctx0, q_cls, 0, 2, 1, 3); // -> [d_head, 1, n_head]
                ggml_tensor * k_perm = ggml_permute(ctx0, Kcur, 0, 2, 1, 3); // -> [d_head, n_pos, n_head]

                // raw scores, all heads, one query -> [n_pos, 1, n_head]
                ggml_tensor * scores = ggml_mul_mat(ctx0, k_perm, q_cls);
                // softmax over all n_pos keys (CLS included), matching FasterVLM's
                // attentions[-2][:, :, 0, :] before its [1:] slice below
                scores = ggml_soft_max_ext(ctx0, scores, nullptr, kq_scale, 0.0f);

                // drop the CLS-self entry (key position 0) post-softmax; pre-softmax
                // exclusion would rescale each head's remaining probs by a different
                // per-head constant and silently change the cross-head-averaged ranking
                ggml_tensor * scores_patches = ggml_view_3d(ctx0, scores, n_patches, 1, n_head,
                    scores->nb[1], scores->nb[2], scores->nb[0]); // -> [n_patches, 1, n_head]

                // mean over heads: bring n_head to axis 0 (ggml_mean/ggml_sum_rows
                // require unit stride on the reduced axis, hence the ggml_cont)
                ggml_tensor * scores_by_head = ggml_cont(ctx0, ggml_permute(ctx0, scores_patches, 1, 2, 0, 3));
                cls_scores = ggml_mean(ctx0, scores_by_head); // -> [1, n_patches]
                cls_scores = ggml_reshape_1d(ctx0, cls_scores, n_patches);
                cb(cls_scores, "cls_scores", il);
            }
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL, eps, il);
        cb(cur, "ffn_inp_normed", il);

        // ffn
        cur = build_ffn(cur,
            layer.ff_up_w, layer.ff_up_b,
            layer.ff_gate_w, layer.ff_gate_b,
            layer.ff_down_w, layer.ff_down_b,
            hparams.ffn_op, il);

        cb(cur, "ffn_out", il);

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(cur, "layer_out", il);

        inpL = cur;
    }

    // post-layernorm
    if (model.post_ln_w) {
        inpL = build_norm(inpL, model.post_ln_w, model.post_ln_b, NORM_TYPE_NORMAL, eps, -1);
    }

    ggml_tensor * embeddings = inpL;

    // process vision feature layers (used by granite)
    {
        // final layer is a vision feature layer
        if (hparams.is_feature_layer(max_feature_layer)) {
            embedding_stack.push_back(inpL);
        }

        // If feature layers are explicitly set, stack them (if we have multiple)
        if (!embedding_stack.empty()) {
            embeddings = embedding_stack[0];
            for (size_t i = 1; i < embedding_stack.size(); i++) {
                embeddings = ggml_concat(ctx0, embeddings, embedding_stack[i], 0);
            }
        }
    }

    // llava projector (also used by granite)
    if (hparams.has_llava_projector) {
        embeddings = ggml_reshape_2d(ctx0, embeddings, embeddings->ne[0], embeddings->ne[1]);

        if (prune_visual_tokens) {
            // Visual token pruning: top-K patches by mean CLS-attention score,
            // replacing the "patches" input gather below. clip_n_output_tokens
            // (clip.cpp) computes the matching K, and clip_image_batch_encode
            // (clip.cpp) skips the "patches" input fill under the same condition
            // as prune_visual_tokens above -- all three must stay in agreement.
            GGML_ASSERT(cls_scores != nullptr);
            const int K = std::max(1, (int) std::round(n_patches * hparams.visual_keep));

            // top-K patch-space indices (0..n_patches-1), descending by score
            ggml_tensor * kept_desc = ggml_argsort_top_k(ctx0, cls_scores, K);

            // restore spatial (ascending patch-index) order within the kept set
            ggml_tensor * kept_desc_f32 = ggml_cast(ctx0, kept_desc, GGML_TYPE_F32);
            ggml_tensor * perm = ggml_argsort(ctx0, kept_desc_f32, GGML_SORT_ORDER_ASC);

            // row-offset view skipping the CLS row (row 0); patch i lives at row i+1
            ggml_tensor * patch_rows = ggml_view_2d(ctx0, embeddings, n_embd, n_patches,
                embeddings->nb[1], embeddings->nb[1]);

            ggml_tensor * picked = ggml_get_rows(ctx0, patch_rows, kept_desc); // [n_embd, K], score order
            embeddings = ggml_get_rows(ctx0, picked, perm); // [n_embd, K], ascending spatial order
        } else {
            ggml_tensor * patches = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
            ggml_set_name(patches, "patches");
            ggml_set_input(patches);

            // shape [1, 576, 1024]
            // ne is whcn, ne = [1024, 576, 1, 1]
            embeddings = ggml_get_rows(ctx0, embeddings, patches);
        }

        // print_tensor_info(embeddings, "embeddings");

        // llava projector
        if (proj_type == PROJECTOR_TYPE_MLP) {
            embeddings = build_mm(model.mm_0_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);

            embeddings = ggml_gelu(ctx0, embeddings);
            if (model.mm_2_w) {
                embeddings = build_mm(model.mm_2_w, embeddings);
                embeddings = ggml_add(ctx0, embeddings, model.mm_2_b);
            }
        }
        else if (proj_type == PROJECTOR_TYPE_MLP_NORM) {
            embeddings = build_mm(model.mm_0_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_0_b);
            // ggml_tensor_printf(embeddings, "mm_0_w",0,true,false);
            // First LayerNorm
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_1_w),
                                model.mm_1_b);

            // GELU activation
            embeddings = ggml_gelu(ctx0, embeddings);

            // Second linear layer
            embeddings = build_mm(model.mm_3_w, embeddings);
            embeddings = ggml_add(ctx0, embeddings, model.mm_3_b);

            // Second LayerNorm
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_4_w),
                                model.mm_4_b);
        }
        else if (proj_type == PROJECTOR_TYPE_LDP) {
            // MobileVLM projector
            int n_patch = 24;
            ggml_tensor * mlp_1 = build_mm(model.mm_model_mlp_1_w, embeddings);
            mlp_1 = ggml_add(ctx0, mlp_1, model.mm_model_mlp_1_b);
            mlp_1 = ggml_gelu(ctx0, mlp_1);
            ggml_tensor * mlp_3 = build_mm(model.mm_model_mlp_3_w, mlp_1);
            mlp_3 = ggml_add(ctx0, mlp_3, model.mm_model_mlp_3_b);
            // mlp_3 shape = [1, 576, 2048], ne = [2048, 576, 1, 1]

            // block 1
            ggml_tensor * block_1 = nullptr;
            {
                // transpose from [1, 576, 2048] --> [1, 2048, 576] --> [1, 2048, 24, 24]
                mlp_3 = ggml_permute(ctx0, mlp_3, 1, 0, 2, 3);
                mlp_3 = ggml_cont_4d(ctx0, mlp_3, n_patch, n_patch, mlp_3->ne[1], mlp_3->ne[2]);
                // stride = 1, padding = 1, bias is nullptr
                block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_1_block_0_0_w, mlp_3, 1, 1, 1, 1, 1, 1);

                // layer norm
                // // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_0_1_w), model.mm_model_block_1_block_0_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));

                // block_1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                // hardswish
                ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                // pointwise conv
                block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                block_1 = build_mm(model.mm_model_block_1_block_1_fc1_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc1_b);
                block_1 = ggml_relu(ctx0, block_1);
                block_1 = build_mm(model.mm_model_block_1_block_1_fc2_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_1_block_1_fc2_b);
                block_1 = ggml_hardsigmoid(ctx0, block_1);
                // block_1_hw shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1], block_1 shape = [1, 2048], ne = [2048, 1, 1, 1]
                block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                int w = block_1->ne[0], h = block_1->ne[1];
                block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));

                // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                block_1 = build_mm(model.mm_model_block_1_block_2_0_w, block_1);
                block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);

                // block_1 shape = [1, 24, 24, 2048], ne = [2048, 24, 24, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_1_block_2_1_w), model.mm_model_block_1_block_2_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // block1 shape = [1, 2048, 24, 24], ne = [24, 24, 2048, 1]
                // residual
                block_1 = ggml_add(ctx0, mlp_3, block_1);
            }

            // block_2
            {
                // stride = 2
                block_1 = ggml_conv_2d_dw(ctx0, model.mm_model_block_2_block_0_0_w, block_1, 2, 2, 1, 1, 1, 1);

                // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                // layer norm
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 2, 0, 3));
                // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_0_1_w), model.mm_model_block_2_block_0_1_b);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 2, 0, 1, 3));
                // block_1 shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1]
                // hardswish
                ggml_tensor * block_1_hw = ggml_hardswish(ctx0, block_1);

                // not sure the parameters is right for globalAvgPooling
                block_1 = ggml_pool_2d(ctx0, block_1_hw, GGML_OP_POOL_AVG, block_1_hw->ne[0], block_1_hw->ne[1], block_1_hw->ne[0], block_1_hw->ne[1], 0, 0);
                // block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                // pointwise conv
                block_1 = ggml_reshape_2d(ctx0, block_1, block_1->ne[0]*block_1->ne[1]*block_1->ne[2], block_1->ne[3]);
                block_1 = build_mm(model.mm_model_block_2_block_1_fc1_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc1_b);
                block_1 = ggml_relu(ctx0, block_1);
                block_1 = build_mm(model.mm_model_block_2_block_1_fc2_w, block_1);
                block_1 = ggml_add(ctx0, block_1, model.mm_model_block_2_block_1_fc2_b);
                block_1 = ggml_hardsigmoid(ctx0, block_1);

                // block_1_hw shape = [1, 2048, 12, 12], ne = [12, 12, 2048, 1], block_1 shape = [1, 2048, 1, 1], ne = [1, 1, 2048, 1]
                block_1 = ggml_reshape_4d(ctx0, block_1, 1, 1, block_1->ne[0], block_1->ne[1]);
                block_1 = ggml_mul(ctx0, block_1_hw, block_1);

                int w = block_1->ne[0], h = block_1->ne[1];
                block_1 = ggml_reshape_3d(ctx0, block_1, w*h, block_1->ne[2], block_1->ne[3]);
                block_1 = ggml_cont(ctx0, ggml_permute(ctx0, block_1, 1, 0, 2, 3));
                // block_1 shape = [1, 24*24, 2048], ne = [24*24, 2048, 1]
                block_1 = build_mm(model.mm_model_block_2_block_2_0_w, block_1);
                block_1 = ggml_reshape_4d(ctx0, block_1, block_1->ne[0], w, h, block_1->ne[3]);


                // block_1 shape = [1, 12, 12, 2048], ne = [2048, 12, 12, 1]
                block_1 = ggml_norm(ctx0, block_1, eps);
                block_1 = ggml_add(ctx0, ggml_mul(ctx0, block_1, model.mm_model_block_2_block_2_1_w), model.mm_model_block_2_block_2_1_b);
                block_1 = ggml_reshape_3d(ctx0, block_1, block_1->ne[0], block_1->ne[1] * block_1->ne[2], block_1->ne[3]);
                // block_1 shape = [1, 144, 2048], ne = [2048, 144, 1]
            }
            embeddings = block_1;
        }
        else if (proj_type == PROJECTOR_TYPE_LDPV2)
        {
            int n_patch = 24;
            ggml_tensor * mlp_0 = build_mm(model.mm_model_mlp_0_w, embeddings);
            mlp_0 = ggml_add(ctx0, mlp_0, model.mm_model_mlp_0_b);
            mlp_0 = ggml_gelu(ctx0, mlp_0);
            ggml_tensor * mlp_2 = build_mm(model.mm_model_mlp_2_w, mlp_0);
            mlp_2 = ggml_add(ctx0, mlp_2, model.mm_model_mlp_2_b);
            // mlp_2 ne = [2048, 576, 1, 1]
            // // AVG Pool Layer 2*2, strides = 2
            mlp_2 = ggml_permute(ctx0, mlp_2, 1, 0, 2, 3);
            // mlp_2 ne = [576, 2048, 1, 1]
            mlp_2 = ggml_cont_4d(ctx0, mlp_2, n_patch, n_patch, mlp_2->ne[1], mlp_2->ne[2]);
            // mlp_2 ne [24, 24, 2048, 1]
            mlp_2 = ggml_pool_2d(ctx0, mlp_2, GGML_OP_POOL_AVG, 2, 2, 2, 2, 0, 0);
            // weight ne = [3, 3, 2048, 1]
            ggml_tensor * peg_0 = ggml_conv_2d_dw(ctx0, model.mm_model_peg_0_w, mlp_2, 1, 1, 1, 1, 1, 1);
            peg_0 = ggml_cont(ctx0, ggml_permute(ctx0, peg_0, 1, 2, 0, 3));
            peg_0 = ggml_add(ctx0, peg_0, model.mm_model_peg_0_b);
            mlp_2 = ggml_cont(ctx0, ggml_permute(ctx0, mlp_2, 1, 2, 0, 3));
            peg_0 = ggml_add(ctx0, peg_0, mlp_2);
            peg_0 = ggml_reshape_3d(ctx0, peg_0, peg_0->ne[0], peg_0->ne[1] * peg_0->ne[2], peg_0->ne[3]);
            embeddings = peg_0;
        }
        else {
            GGML_ABORT("fatal error");
        }
    }

    // glm projector
    else if (proj_type == PROJECTOR_TYPE_GLM_EDGE) {
        size_t gridsz = (size_t)sqrt(embeddings->ne[1]);
        embeddings = ggml_permute(ctx0,embeddings,1,0,2,3);
        embeddings = ggml_cont_3d(ctx0, embeddings, gridsz, gridsz, embeddings->ne[1]);
        embeddings = ggml_conv_2d(ctx0, model.mm_model_adapter_conv_w, embeddings, 2, 2, 0, 0, 1, 1);
        embeddings = ggml_reshape_3d(ctx0, embeddings,embeddings->ne[0]*embeddings->ne[1] , embeddings->ne[2], batch_size);
        embeddings = ggml_cont(ctx0, ggml_permute(ctx0,embeddings, 1, 0, 2, 3));
        embeddings = ggml_add(ctx0, embeddings, model.mm_model_adapter_conv_b);
        // GLU
        {
            embeddings = build_mm(model.mm_model_mlp_0_w, embeddings);
            embeddings = ggml_norm(ctx0, embeddings, eps);
            embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, model.mm_model_ln_q_w), model.mm_model_ln_q_b);
            embeddings = ggml_gelu_inplace(ctx0, embeddings);
            ggml_tensor * x = embeddings;
            embeddings = build_mm(model.mm_model_mlp_2_w, embeddings);
            x = build_mm(model.mm_model_mlp_1_w,x);
            embeddings = ggml_swiglu_split(ctx0, embeddings, x);
            embeddings = build_mm(model.mm_model_mlp_3_w, embeddings);
        }
        // arrangement of BOI/EOI token embeddings
        // note: these embeddings are not present in text model, hence we cannot process them as text tokens
        // see: https://huggingface.co/THUDM/glm-edge-v-2b/blob/main/siglip.py#L53
        {
            embeddings = ggml_concat(ctx0, model.mm_boi, embeddings, 1); // BOI
            embeddings = ggml_concat(ctx0, embeddings, model.mm_eoi, 1); // EOI
        }
    }

    else {
        GGML_ABORT("llava: unknown projector type");
    }

    // build the graph
    ggml_build_forward_expand(gf, embeddings);

    return gf;
}
