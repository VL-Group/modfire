# Schema Docs

- [1. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `comments`](#comments)
- [2. ![Required](https://img.shields.io/badge/Required-blue) Property `model`](#model)
  - [2.1. ![Required](https://img.shields.io/badge/Required-blue) Property `key`](#model_key)
  - [2.2. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `params`](#model_params)
    - [2.2.1. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `additionalProperties`](#model_params_additionalProperties)
  - [2.3. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `temperature`](#model_temperature)
    - [2.3.1. ![Required](https://img.shields.io/badge/Required-blue) Property `key`](#model_temperature_key)
    - [2.3.2. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `params`](#model_temperature_params)
      - [2.3.2.1. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `additionalProperties`](#model_temperature_params_additionalProperties)
- [3. ![Required](https://img.shields.io/badge/Required-blue) Property `train`](#train)
  - [3.1. ![Required](https://img.shields.io/badge/Required-blue) Property `criterion`](#train_criterion)
  - [3.2. ![Required](https://img.shields.io/badge/Required-blue) Property `database`](#train_database)
    - [3.2.1. ![Required](https://img.shields.io/badge/Required-blue) Property `key`](#train_database_key)
    - [3.2.2. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `params`](#train_database_params)
      - [3.2.2.1. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `additionalProperties`](#train_database_params_additionalProperties)
    - [3.2.3. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `pipeline`](#train_database_pipeline)
  - [3.3. ![Required](https://img.shields.io/badge/Required-blue) Property `earlyStop`](#train_earlyStop)
  - [3.4. ![Required](https://img.shields.io/badge/Required-blue) Property `epoch`](#train_epoch)
  - [3.5. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `externalLib`](#train_externalLib)
    - [3.5.1. items](#autogenerated_heading_2)
  - [3.6. ![Required](https://img.shields.io/badge/Required-blue) Property `gpu`](#train_gpu)
    - [3.6.1. ![Required](https://img.shields.io/badge/Required-blue) Property `gpus`](#train_gpu_gpus)
    - [3.6.2. ![Required](https://img.shields.io/badge/Required-blue) Property `vRam`](#train_gpu_vRam)
    - [3.6.3. ![Required](https://img.shields.io/badge/Required-blue) Property `wantsMore`](#train_gpu_wantsMore)
  - [3.7. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `hooks`](#train_hooks)
    - [3.7.1. items](#autogenerated_heading_3)
  - [3.8. ![Required](https://img.shields.io/badge/Required-blue) Property `numReturns`](#train_numReturns)
  - [3.9. ![Required](https://img.shields.io/badge/Required-blue) Property `optim`](#train_optim)
  - [3.10. ![Required](https://img.shields.io/badge/Required-blue) Property `querySet`](#train_querySet)
  - [3.11. ![Required](https://img.shields.io/badge/Required-blue) Property `saveDir`](#train_saveDir)
  - [3.12. ![Required](https://img.shields.io/badge/Required-blue) Property `schdr`](#train_schdr)
  - [3.13. ![Required](https://img.shields.io/badge/Required-blue) Property `trainSet`](#train_trainSet)
  - [3.14. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `trainer`](#train_trainer)
  - [3.15. ![Required](https://img.shields.io/badge/Required-blue) Property `valFreq`](#train_valFreq)

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Defined in**            | #/definitions/ConfigSchema                                                                               |

| Property                 | Pattern | Type   | Deprecated | Definition                   | Title/Description                                                                  |
| ------------------------ | ------- | ------ | ---------- | ---------------------------- | ---------------------------------------------------------------------------------- |
| - [comments](#comments ) | No      | string | No         | -                            | comments                                                                           |
| + [model](#model )       | No      | object | No         | In #/definitions/ModelSchema | Model to use. Avaliable params are e.g. \`backbone\`, \`bits\` and \`hashMethod\`. |
| + [train](#train )       | No      | object | No         | In #/definitions/TrainSchema | Training configs.                                                                  |

## <a name="comments"></a>1. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `comments`

**Title:** comments

|             |          |
| ----------- | -------- |
| **Type**    | `string` |
| **Default** | `""`     |

**Description:** Optional comments to describe this config.

## <a name="model"></a>2. ![Required](https://img.shields.io/badge/Required-blue) Property `model`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Defined in**            | #/definitions/ModelSchema                                                                                |

**Description:** Model to use. Avaliable params are e.g. `backbone`, `bits` and `hashMethod`.

| Property                             | Pattern | Type   | Deprecated | Definition                     | Title/Description                        |
| ------------------------------------ | ------- | ------ | ---------- | ------------------------------ | ---------------------------------------- |
| + [key](#model_key )                 | No      | string | No         | -                              | key                                      |
| - [params](#model_params )           | No      | object | No         | -                              | params                                   |
| - [temperature](#model_temperature ) | No      | object | No         | In #/definitions/GeneralSchema | A spec of temperature tuning schdeduler. |

### <a name="model_key"></a>2.1. ![Required](https://img.shields.io/badge/Required-blue) Property `key`

**Title:** key

|          |          |
| -------- | -------- |
| **Type** | `string` |

**Description:** A unique key used to retrieve in ModelRegistry.

### <a name="model_params"></a>2.2. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `params`

**Title:** params

|                           |                                                                                                                                                                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                                                                  |
| **Additional properties** | [![Should-conform](https://img.shields.io/badge/Should-conform-blue)](#model_params_additionalProperties "Each additional property must conform to the following schema") |
| **Default**               | `{}`                                                                                                                                                                      |

**Description:** Corresponding funcation call parameters. So the whole call is `registry.get(key)(**params)`.

| Property                                                      | Pattern | Type   | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [additionalProperties](#model_params_additionalProperties ) | No      | object | No         | -          | -                 |

#### <a name="model_params_additionalProperties"></a>2.2.1. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `additionalProperties`

|                           |                                                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                          |
| **Additional properties** | [![Any type: allowed](https://img.shields.io/badge/Any%20type-allowed-green)](# "Additional Properties of any type are allowed.") |

### <a name="model_temperature"></a>2.3. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `temperature`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Defined in**            | #/definitions/GeneralSchema                                                                              |

**Description:** A spec of temperature tuning schdeduler.

| Property                               | Pattern | Type   | Deprecated | Definition | Title/Description |
| -------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| + [key](#model_temperature_key )       | No      | string | No         | -          | key               |
| - [params](#model_temperature_params ) | No      | object | No         | -          | params            |

#### <a name="model_temperature_key"></a>2.3.1. ![Required](https://img.shields.io/badge/Required-blue) Property `key`

**Title:** key

|             |          |
| ----------- | -------- |
| **Type**    | `string` |
| **Default** | `null`   |

**Description:** A unique key used to retrieve in registry. For example, given `Lamb` for optimizers, it will check `OptimRegistry` and find the optimizer `apex.optim.FusedLAMB`.

#### <a name="model_temperature_params"></a>2.3.2. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `params`

**Title:** params

|                           |                                                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                                                                              |
| **Additional properties** | [![Should-conform](https://img.shields.io/badge/Should-conform-blue)](#model_temperature_params_additionalProperties "Each additional property must conform to the following schema") |
| **Default**               | `{}`                                                                                                                                                                                  |

**Description:** Corresponding funcation call parameters. So the whole call is `registry.get(key)(**params)`.

| Property                                                                  | Pattern | Type   | Deprecated | Definition | Title/Description |
| ------------------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [additionalProperties](#model_temperature_params_additionalProperties ) | No      | object | No         | -          | -                 |

##### <a name="model_temperature_params_additionalProperties"></a>2.3.2.1. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `additionalProperties`

|                           |                                                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                          |
| **Additional properties** | [![Any type: allowed](https://img.shields.io/badge/Any%20type-allowed-green)](# "Additional Properties of any type are allowed.") |

## <a name="train"></a>3. ![Required](https://img.shields.io/badge/Required-blue) Property `train`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Defined in**            | #/definitions/TrainSchema                                                                                |

**Description:** Training configs.

| Property                             | Pattern | Type                    | Deprecated | Definition                                 | Title/Description                                                                                                                                                                                   |
| ------------------------------------ | ------- | ----------------------- | ---------- | ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| + [criterion](#train_criterion )     | No      | object                  | No         | Same as [temperature](#model_temperature ) | Loss function used for training.                                                                                                                                                                    |
| + [database](#train_database )       | No      | object                  | No         | In #/definitions/DatasetSchema             | A spec to load images per line for evalution database.                                                                                                                                              |
| + [earlyStop](#train_earlyStop )     | No      | integer                 | No         | -                                          | earlyStop                                                                                                                                                                                           |
| + [epoch](#train_epoch )             | No      | integer                 | No         | -                                          | epoch                                                                                                                                                                                               |
| - [externalLib](#train_externalLib ) | No      | array of string or null | No         | -                                          | externalLib                                                                                                                                                                                         |
| + [gpu](#train_gpu )                 | No      | object                  | No         | In #/definitions/GPUSchema                 | GPU configs for training.                                                                                                                                                                           |
| - [hooks](#train_hooks )             | No      | array of object or null | No         | -                                          | hooks                                                                                                                                                                                               |
| + [numReturns](#train_numReturns )   | No      | integer                 | No         | -                                          | numReturns                                                                                                                                                                                          |
| + [optim](#train_optim )             | No      | object                  | No         | Same as [temperature](#model_temperature ) | Optimizer used for training. As for current we have \`Adam\` and \`Lamb\`.                                                                                                                          |
| + [querySet](#train_querySet )       | No      | object                  | No         | Same as [database](#train_database )       | A spec to load images per line for evalution query.                                                                                                                                                 |
| + [saveDir](#train_saveDir )         | No      | string                  | No         | -                                          | saveDir                                                                                                                                                                                             |
| + [schdr](#train_schdr )             | No      | object                  | No         | Same as [temperature](#model_temperature ) | Learning rate scheduler used for training. As for current we have \`ReduceLROnPlateau\`, \`Exponential\`, \`MultiStep\`, \`OneCycle\` and all schedulers defined in \`modfire.train.lrSchedulers\`. |
| + [trainSet](#train_trainSet )       | No      | object                  | No         | Same as [database](#train_database )       | A spec to load images per line for training.                                                                                                                                                        |
| - [trainer](#train_trainer )         | No      | string                  | No         | -                                          | trainer                                                                                                                                                                                             |
| + [valFreq](#train_valFreq )         | No      | integer                 | No         | -                                          | valFreq                                                                                                                                                                                             |

### <a name="train_criterion"></a>3.1. ![Required](https://img.shields.io/badge/Required-blue) Property `criterion`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Same definition as**    | [temperature](#model_temperature)                                                                        |

**Description:** Loss function used for training.

### <a name="train_database"></a>3.2. ![Required](https://img.shields.io/badge/Required-blue) Property `database`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Defined in**            | #/definitions/DatasetSchema                                                                              |

**Description:** A spec to load images per line for evalution database.

| Property                                | Pattern | Type   | Deprecated | Definition                                 | Title/Description                |
| --------------------------------------- | ------- | ------ | ---------- | ------------------------------------------ | -------------------------------- |
| + [key](#train_database_key )           | No      | string | No         | -                                          | key                              |
| - [params](#train_database_params )     | No      | object | No         | -                                          | params                           |
| - [pipeline](#train_database_pipeline ) | No      | object | No         | Same as [temperature](#model_temperature ) | A spec of data loading pipeline. |

#### <a name="train_database_key"></a>3.2.1. ![Required](https://img.shields.io/badge/Required-blue) Property `key`

**Title:** key

|          |          |
| -------- | -------- |
| **Type** | `string` |

**Description:** A unique key used to retrieve in DatasetRegistry.

#### <a name="train_database_params"></a>3.2.2. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `params`

**Title:** params

|                           |                                                                                                                                                                                    |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                                                                           |
| **Additional properties** | [![Should-conform](https://img.shields.io/badge/Should-conform-blue)](#train_database_params_additionalProperties "Each additional property must conform to the following schema") |
| **Default**               | `{}`                                                                                                                                                                               |

**Description:** Corresponding funcation call parameters. So the whole call is `registry.get(key)(**params)`.

| Property                                                               | Pattern | Type   | Deprecated | Definition | Title/Description |
| ---------------------------------------------------------------------- | ------- | ------ | ---------- | ---------- | ----------------- |
| - [additionalProperties](#train_database_params_additionalProperties ) | No      | object | No         | -          | -                 |

##### <a name="train_database_params_additionalProperties"></a>3.2.2.1. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `additionalProperties`

|                           |                                                                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                                          |
| **Additional properties** | [![Any type: allowed](https://img.shields.io/badge/Any%20type-allowed-green)](# "Additional Properties of any type are allowed.") |

#### <a name="train_database_pipeline"></a>3.2.3. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `pipeline`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Same definition as**    | [temperature](#model_temperature)                                                                        |

**Description:** A spec of data loading pipeline.

### <a name="train_earlyStop"></a>3.3. ![Required](https://img.shields.io/badge/Required-blue) Property `earlyStop`

**Title:** earlyStop

|          |           |
| -------- | --------- |
| **Type** | `integer` |

**Description:** Early stop after how many evaluations.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

### <a name="train_epoch"></a>3.4. ![Required](https://img.shields.io/badge/Required-blue) Property `epoch`

**Title:** epoch

|          |           |
| -------- | --------- |
| **Type** | `integer` |

**Description:** Total training epochs.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

### <a name="train_externalLib"></a>3.5. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `externalLib`

**Title:** externalLib

|             |                           |
| ----------- | ------------------------- |
| **Type**    | `array of string or null` |
| **Default** | `[]`                      |

**Description:** External libraries used for training. All python files in `externalLib` will be imported as modules. In this way, you could extend registries.

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be         | Description |
| --------------------------------------- | ----------- |
| [externalLib](#train_externalLib_items) | -           |

#### <a name="autogenerated_heading_2"></a>3.5.1. items

**Title:** externalLib

|          |          |
| -------- | -------- |
| **Type** | `string` |

### <a name="train_gpu"></a>3.6. ![Required](https://img.shields.io/badge/Required-blue) Property `gpu`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Defined in**            | #/definitions/GPUSchema                                                                                  |

**Description:** GPU configs for training.

| Property                             | Pattern | Type    | Deprecated | Definition | Title/Description |
| ------------------------------------ | ------- | ------- | ---------- | ---------- | ----------------- |
| + [gpus](#train_gpu_gpus )           | No      | integer | No         | -          | gpus              |
| + [vRam](#train_gpu_vRam )           | No      | integer | No         | -          | vRam              |
| + [wantsMore](#train_gpu_wantsMore ) | No      | boolean | No         | -          | wantsMore         |

#### <a name="train_gpu_gpus"></a>3.6.1. ![Required](https://img.shields.io/badge/Required-blue) Property `gpus`

**Title:** gpus

|          |           |
| -------- | --------- |
| **Type** | `integer` |

**Description:** Number of gpus for training. This affects the `world size` of PyTorch DDP.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

#### <a name="train_gpu_vRam"></a>3.6.2. ![Required](https://img.shields.io/badge/Required-blue) Property `vRam`

**Title:** vRam

|          |           |
| -------- | --------- |
| **Type** | `integer` |

**Description:** Minimum VRam required for each gpu. Set it to `-1` to use all gpus.

#### <a name="train_gpu_wantsMore"></a>3.6.3. ![Required](https://img.shields.io/badge/Required-blue) Property `wantsMore`

**Title:** wantsMore

|          |           |
| -------- | --------- |
| **Type** | `boolean` |

**Description:** Set to `true` to use all visible gpus and all VRams and ignore `gpus` and `vRam`.

### <a name="train_hooks"></a>3.7. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `hooks`

**Title:** hooks

|             |                           |
| ----------- | ------------------------- |
| **Type**    | `array of object or null` |
| **Default** | `[]`                      |

**Description:** Hooks used for training. Key is used to retrieve hook from `LBHash.train.hooks`.

|                      | Array restrictions |
| -------------------- | ------------------ |
| **Min items**        | N/A                |
| **Max items**        | N/A                |
| **Items unicity**    | False              |
| **Additional items** | False              |
| **Tuple validation** | See below          |

| Each item of this array must be     | Description |
| ----------------------------------- | ----------- |
| [GeneralSchema](#train_hooks_items) | -           |

#### <a name="autogenerated_heading_3"></a>3.7.1. items

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Same definition as**    | [temperature](#model_temperature)                                                                        |

### <a name="train_numReturns"></a>3.8. ![Required](https://img.shields.io/badge/Required-blue) Property `numReturns`

**Title:** numReturns

|          |           |
| -------- | --------- |
| **Type** | `integer` |

**Description:** Rank list return number of samples.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

### <a name="train_optim"></a>3.9. ![Required](https://img.shields.io/badge/Required-blue) Property `optim`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Same definition as**    | [temperature](#model_temperature)                                                                        |

**Description:** Optimizer used for training. As for current we have `Adam` and `Lamb`.

### <a name="train_querySet"></a>3.10. ![Required](https://img.shields.io/badge/Required-blue) Property `querySet`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Same definition as**    | [database](#train_database)                                                                              |

**Description:** A spec to load images per line for evalution query.

### <a name="train_saveDir"></a>3.11. ![Required](https://img.shields.io/badge/Required-blue) Property `saveDir`

**Title:** saveDir

|          |          |
| -------- | -------- |
| **Type** | `string` |

**Description:** A dir path to save model checkpoints, TensorBoard messages and logs.

### <a name="train_schdr"></a>3.12. ![Required](https://img.shields.io/badge/Required-blue) Property `schdr`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Same definition as**    | [temperature](#model_temperature)                                                                        |

**Description:** Learning rate scheduler used for training. As for current we have `ReduceLROnPlateau`, `Exponential`, `MultiStep`, `OneCycle` and all schedulers defined in `modfire.train.lrSchedulers`.

### <a name="train_trainSet"></a>3.13. ![Required](https://img.shields.io/badge/Required-blue) Property `trainSet`

|                           |                                                                                                          |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Type**                  | `object`                                                                                                 |
| **Additional properties** | [![Not allowed](https://img.shields.io/badge/Not%20allowed-red)](# "Additional Properties not allowed.") |
| **Same definition as**    | [database](#train_database)                                                                              |

**Description:** A spec to load images per line for training.

### <a name="train_trainer"></a>3.14. ![Optional](https://img.shields.io/badge/Optional-yellow) Property `trainer`

**Title:** trainer

|             |                 |
| ----------- | --------------- |
| **Type**    | `string`        |
| **Default** | `"BaseTrainer"` |

**Description:** A key to retrieve from TrainerBuilder, default is `BaseTrainer`.

### <a name="train_valFreq"></a>3.15. ![Required](https://img.shields.io/badge/Required-blue) Property `valFreq`

**Title:** valFreq

|          |           |
| -------- | --------- |
| **Type** | `integer` |

**Description:** Run validation after every `valFreq` epochs.

| Restrictions |        |
| ------------ | ------ |
| **Minimum**  | &gt; 0 |

----------------------------------------------------------------------------------------------------------------------------
Generated using [json-schema-for-humans](https://github.com/coveooss/json-schema-for-humans)
