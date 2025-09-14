# Project Codes

This project integrates robotic control, physiological signal processing, and machine learning to analyze participant behavior and biometric responses. It involves three main components: Unity, FactorizePhys, and AutoGluon.

---

## Unity Integration

### `PickAndPlace.cs`
Implements a pick-and-place task using the Niryo-One robot arm in Unity. This is the main script for the experiment design.

---

## Codes with FactorizePhys

### `mp4_to_npy.py`
Converts participant `.mp4` videos into `.npy` format for use with [FactorizePhys](https://github.com/PhysiologicAILab/FactorizePhys).

### `Inference.py`
Performs model inference using FactorizePhys to generate physiological weights.  
> Requires FactorizePhys. Refer to the [official repository](https://github.com/PhysiologicAILab/FactorizePhys) for setup instructions.

### `Visualise.py`
Generates visualizations of participants' heart rate using rPPG signals, as presented in the final report.

### `VisualiseProcess.py`
Displays raw pixel intensity over time across RGB channels and computes temporal differences in the red channel to highlight signal dynamics.

---

## AutoGluon Analysis 
> Need to download AutoGluon

### Label Mapping

Each experimental condition is assigned a unique numerical label used for training the AutoGluon MultiModalPredictor model:

| Condition         | Label ID |
|------------------|----------|
| Perfect          | 0        |
| DelayAfterReach  | 1        |
| DelayBeforeRelease | 2      |
| StartDelay       | 3        |
| Shake            | 4        |
| ArcMovement      | 5        |
| VariableSpeed    | 6        |
| PartialGrasp     | 7        |
| Obvious          | 8        |


### `step5_train.py`
Trains an AutoGluon model using frame-level video data to classify and predict error types in participant behavior.

### `step5_predict.py`
Optional utility script for visualizing selected prediction parameters. Useful for debugging or supplementary analysis.

---

## Notes

- Ensure all dependencies for Unity, FactorizePhys, and AutoGluon are properly installed.
- Recommended Python version: 3.8+
- For FactorizePhys setup, follow instructions in their [GitHub repository](https://github.com/PhysiologicAILab/FactorizePhys).