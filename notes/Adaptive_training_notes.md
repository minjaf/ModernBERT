Here is where we set training methods in trainer.py:

```python
    def _eval_train_metrics(self, device_batch):
        assert self._train_data_spec is not None, 'The train data spec should be set on __init__ or fit()'
        assert self.state.train_metrics is not None, 'The train metrics should be set on __init__ or fit()'
        # We disable FP8 autocast in eval metrics and default to the activation dtype for the forward pass
        # This is because FP8 in TE requires all eval data sizes to be divisible by 16 which does not hold for all evaluation datasets.
        # See https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html for more info.
        # Note: the activation dtype is BF16 if FSDP Mixed Precision PURE is enabled and FP32 if FSDP Mixed Precision FULL is enabled.
        # See https://github.com/NVIDIA/TransformerEngine/blob/8e039fdcd98fc56582d81e373880c1509c2b8f73/transformer_engine/pytorch/module/linear.py#L250-L252 and \
        # https://github.com/NVIDIA/TransformerEngine/blob/8e039fdcd98fc56582d81e373880c1509c2b8f73/transformer_engine/pytorch/module/base.py#L495-L513 for more info.
        with torch.no_grad(),\
                model_eval_mode(self.state.model),\
                _get_precision_context(self.state.precision, self.state.precision_config, self.state.deepspeed_enabled, fp8_autocast_enabled=False):
            eval_outputs = self._original_model.eval_forward(device_batch, self.state.outputs)
            for metric in self.state.train_metrics.values():
                self._original_model.update_metric(
                    device_batch,
                    eval_outputs,
                    metric,
                )
```

And here we set the outputs:

```python
    def _train_microbatch(...)
			...
            with _get_precision_context(
                self.state.precision,
                self.state.precision_config,
                self.state.deepspeed_enabled,
            ):
                self.state.outputs = self.state.model(self.state.batch)
```

Here we define set of metrics:

```python
The metrics used are defined in your model's ``get_metrics()`` method. For more information,
see :doc:`/trainer/evaluation`.
```

## Key modifications to implement adaptive MLM:

1. **Dataset reads MLM probs** (`src/text_data.py`)
2. **Dataset propagates data required to save MLM probs** (`src/text_data.py`)
3. **Custom collate function** (`src/text_data.py`) to handle these additional inputs
4. **The same collate function re-implements masking** to account for MLM masking (`src/text_data.py`)
5. **Metrics class** that stores model prediction and flushes them to file when needed (`src/flex_bert.py`)
6. **EfficientHuggingFaceModel modified** to handle this metrics (`src/flex_bert.py`)

# Writing to file.

a1)
```
for i in range(num_repeat):
    # update only if true prob is > write2file_threshold to save time
    if true_probs[idx_of_token_in_concatenated_batch] > self.write2file_threshold:
        st = offset_starts[idx_of_token_in_concatenated_batch]
        en = offset_ends[idx_of_token_in_concatenated_batch]
        f[str(shard_sample_id)][st:en] = 1. - true_probs[idx_of_token_in_concatenated_batch]
    idx_of_token_in_concatenated_batch += 1
```
a2)
```
data = f[str(shard_sample_id)][:] # read all data
for i in range(num_repeat):
    # update only if true prob is > write2file_threshold, to save time
    if true_probs[idx_of_token_in_concatenated_batch] > self.write2file_threshold:
        st = offset_starts[idx_of_token_in_concatenated_batch]
        en = offset_ends[idx_of_token_in_concatenated_batch]
        data[st:en] = 1. - true_probs[idx_of_token_in_concatenated_batch] # update in memory
    idx_of_token_in_concatenated_batch += 1
f[str(shard_sample_id)][:] = data # single write operation
```

a1) write all values, replace each in file  --> ~3-4k msec


a2) write vals with gt_prob>0.5, replace each in file --> ~400-600 msec


b1) write all values, read all to mem, replace 'in mem', flush all to file --> 80 msec


b2) write val with gt_prob>0.5, read all to mem, replace 'in mem', flush all to file --> 50-60 msec 

# to test code (obtain and visualyze MLM probs), 

1) modify trainer.py (part of composer):

before

```
                total_loss_dict = self._train_batch(use_grad_scaling)

                if use_grad_scaling:
                    self.state.scaler.update()
```

add code:

```
                # # Store initial weight norms
                # wasinit = False 
                # try:
                #     # is initial_weight_norms defined?
                #     initial_weight_norms
                # except NameError:
                #     wasinit = True
                #     initial_weight_norms = {}
                #     with torch.no_grad():
                #         for name, param in self.state.model.named_parameters():
                #             initial_weight_norms[name] = param.data.clone()
                
                # with torch.no_grad():
```

and after, add code:


                # for name, param in self.state.model.named_parameters():
                #     param.data = initial_weight_norms[name].detach()

                # print ("############### DEBUG - AFTER TRAIN BATCH ############### (wasinit = ", wasinit, ")")
                
                # # Check if weights were updated
                # weight_changes = {}
                # with torch.no_grad():
                #     for name, param in self.state.model.named_parameters():
                #         if param.requires_grad and name in initial_weight_norms:
                #             weight_diff = torch.norm(param.data - initial_weight_norms[name])
                #             if weight_diff > 1e-8:  # Small threshold for numerical precision
                #                 weight_changes[name] = weight_diff.item()
                #                 param.data = initial_weight_norms[name].detach()
                
                # if weight_changes:
                #     print(f"WARNING: Weights were updated! Changes detected in {len(weight_changes)} parameters:")
                #     for name, change in weight_changes.items():
                #         print(f"  {name}: {change:.2e}")
                # else:
                #     print("SUCCESS: No weight updates detected - weights are frozen as expected")

Also modify DataCollatorForLanguageModelingWithMLMProbs and set `mask_probs_array` to 1.0 constantly.

Then run with config proceed with `yamls/debug/gena-base-adaptive_mlm_debug.yaml`, let it process ~500 epoches, and proceed with `notebooks/test_MLM_probs.ipynb`. This will generate bedGraphs for IGV.