from menpo.base import MenpoMissingDependencyError

try:
    from menpowidgets import features_selection
except ImportError:
    raise MenpoMissingDependencyError('menpowidgets')


def features_selection_widget():
    r"""
    Widget that allows for easy selection of a features function and its
    options. It also has a 'preview' tab for visual inspection. It returns a
    `list` of length 1 with the selected features function closure.

    Returns
    -------
    features_function : `list` of length ``1``
        The function closure of the features function using `functools.partial`.
        So the function can be called as: ::

            features_image = features_function[0](image)

    Examples
    --------
    The widget can be invoked as ::

        from menpo.feature import features_selection_widget
        features_fun = features_selection_widget()

    And the returned function can be used as ::

        import menpo.io as mio
        image = mio.import_builtin_asset.lenna_png()
        features_image = features_fun[0](image)
    """

    return features_selection()
