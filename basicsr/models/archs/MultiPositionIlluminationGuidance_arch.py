"""Paper-facing name for the RetinexFormer-based model with MS-IFM, D-IGAR, and IGS.

This subclass changes only the configuration-visible class name. Parameter names,
the forward graph, and checkpoint keys remain those of the verified paper model.
"""

from basicsr.models.archs.RetinexFormer_arch import RetinexFormer


class MultiPositionIlluminationGuidance(RetinexFormer):
    """RetinexFormer backbone with the three illumination-guided modules."""
