from captum.attr import (
    IntegratedGradients,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    KernelShap,
    ShapleyValueSampling,
)


bool_multiply_inputs = True
baseline_type = "random"
baseline_type_text = "padding"
dict_method_arguments = {
    "integrated_gradients": {
        "captum_method": IntegratedGradients,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "baseline_type_text": baseline_type_text,
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        # "noback_cudnn": True,
        "batch_size": 8,
    },
    "deeplift": {
        "captum_method": DeepLift,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "baseline_type_text": baseline_type_text,
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        # "noback_cudnn": True,
        "batch_size": 8,
    },
    "deepliftshap": {
        "captum_method": DeepLiftShap,
        "require_baseline": True,
        "baseline_type": "sample",
        "baseline_type_text": "sample",
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        # "noback_cudnn": True,
        "batch_size": 8,
    },
    "gradshap": {
        "captum_method": GradientShap,
        "require_baseline": True,
        "baseline_type": "sample",
        "baseline_type_text": "sample",
        "kwargs_method": {"multiply_by_inputs": bool_multiply_inputs},
        # "noback_cudnn": True,
        "batch_size": 8,
    },
    "kernelshap": {
        "captum_method": KernelShap,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "baseline_type_text": baseline_type_text,
        # "noback_cudnn": False,
        "batch_size": 8,
    },
    "shapleyvalue": {
        "captum_method": ShapleyValueSampling,
        "require_baseline": True,
        "baseline_type": baseline_type,
        "baseline_type_text": baseline_type_text,
        "kwargs_attribution": {"perturbations_per_eval": 32},
        # "noback_cudnn": False,
        "batch_size": 32,
    },
}
