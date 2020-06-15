import pytest
import numpy as np
from scipy.sparse import coo_matrix

from croissant.features import FeatureExtractor as fx


@pytest.mark.parametrize(
    "roi, expected",
    [
        (coo_matrix(np.array([[1, 1, 1],
                              [0, 1, 0]])), 3/2),   # row long axis
        # col long axis, unit dimension
        (coo_matrix(np.array([[1, 0, 0],
                              [1, 0, 0]])), 2),
        (coo_matrix(np.array([[0, 1, 1],
                              [0, 0, 1]])), 1),    # equal axes
    ]
)
def test_ellipticalness(roi, expected):
    assert fx._ellipticalness(roi) == expected


@pytest.mark.parametrize(
    "roi, expected",
    [
        (coo_matrix(np.array([[0.1, 0.2, 0],
                              [0.9, 0, -1.]])), 4),
        (coo_matrix(np.array([[0.0],
                              [0.0]])), 0),
    ],
)
def test_area(roi, expected):
    assert fx._area(roi) == expected


@pytest.mark.parametrize(
    "rois, dff_traces, metadata, expected_ids, expected_rois",
    [
        (
            [   # all ids, Rois input
                {"id": 123, "coo_rows": [0], "coo_cols": [1],
                 "coo_data": [1], "image_shape": (2, 3)},
                {"id": 456, "coo_rows": [1], "coo_cols": [0],
                 "coo_data": [1], "image_shape": (2, 3)},
            ],
            [   # traces
                [1, 1, 1], [2, 2, 2],
            ],
            [   # metadata
                {"depth": 120},
                {"depth": 100},
            ],
            [123, 456],    # expected ids
            [   # expected rois data
                coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
                coo_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
            ]
        ),
        (
            [   # missing ids, Rois input
                {"id": 123, "coo_rows": [0], "coo_cols": [1],
                 "coo_data": [1], "image_shape": (2, 3)},
                {"coo_rows": [1], "coo_cols": [0],
                 "coo_data": [1], "image_shape": (2, 3)},
            ],
            [   # traces
                [1, 1, 1], [2, 2, 2],
            ],
            [   # metadata
                {"depth": 120},
                {"depth": 100},
            ],
            [],    # expected ids
            [   # expected rois data
                coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
                coo_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
            ]
        ),
        (
            [   # coo matrix ROIs
                coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
                coo_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
            ],
            [   # traces
                [1, 1, 1], [2, 2, 2],
            ],
            [   # metadata
                {"depth": 120},
                {"depth": 100},
            ],
            [],    # expected ids
            [   # expected rois data
                coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
                coo_matrix(np.array([[0, 0, 0], [1, 0, 0]]))
            ]
        )
    ]
)
def test_FeatureExtractor_data_inputs(
        rois, dff_traces, metadata, expected_ids, expected_rois):
    """Test overloaded `rois` argument formatting."""
    extractor = fx(rois, dff_traces, metadata)
    assert extractor.roi_ids == expected_ids
    assert len(extractor.rois) == len(expected_rois)
    for e_roi, roi in zip(expected_rois, extractor.rois):
        np.testing.assert_equal(e_roi.toarray(), roi.toarray())


@pytest.mark.parametrize(
    "rois, traces, metadata",
    [
        (   # Unequal trace
            [coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
             coo_matrix(np.array([[0, 1, 0], [0, 0, 0]]))],
            [1],
            [{"rig": "N"}, {"rig": "S"}]
        ),
        (   # Unequal metadata
            [coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
             coo_matrix(np.array([[0, 1, 0], [0, 0, 0]]))],
            [1, 1],
            [{"rig": "N"}]
        ),
        (
            # Unequal ROI
            [coo_matrix(np.array([[0, 1, 0], [0, 0, 0]])),
             coo_matrix(np.array([[0, 1, 0], [0, 0, 0]]))],
            [1, 1],
            [{"rig": "N"}]
        ),
    ]
)
def test_FeatureExtractor_nonequal_input_error(rois, traces, metadata):
    """Test unequal input lengths raise error."""
    with pytest.raises(ValueError):
        fx(rois, traces, metadata)
