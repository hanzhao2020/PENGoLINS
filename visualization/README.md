# Visualization of PENGoLINS results 
For results computed from PENGoLINS, using the Paraview trace script, view_results.py, to generate a Paraview state file or launch an interactive visualization window. This script requires Paraview 5.9.0 or higher, check the pvpython version before running the script:
```bash
pvpython --version
```

Run ``pvpython view_results.py --help`` to see available options. For example, ``start_ind`` and ``end_ind`` are the first and last spline indices for the saved results. ``file_path`` is the location where the results are saved. By default, this script only displays the deformed geometry scaled by ``disp_scale``, whose default value is 1. Users can choose to show the unformed geometry by setting ``show_geom=True`` or von Mises stress (if computed and saved) using ``show_stress=True``. To launch the interactive window, set ``interact=True``.

Running ``pvpython view_results.py`` will generate a Paraview state file by default in the ``states`` directory, the filename is created in the order: "``filename_prefix``\_patch_``start_ind``\_``end_ind``\_disp``disp_scale``.pvsm". Then load the state file into Paraview and click the reset button to view the results. Users can also manually modify the default options, e.g., ``file_path``, in the script and run it in Paraview's Python shell.

Example:

To visualize the displacement field, scaled by a factor of 10, of the Scordelis-Lo roof example:
```bash
pvpython view_results.py --file_path=PATH_TO_RESULTS --end_ind=9 --disp_scale=10 --filename_prefix=ScordelisLo
```

For the visualization of the stress of the eVTOL wing with an opacity of 0.8 and the undeformed geometry with an opacity of 0.4, use the following command:
```bash
pvpython view_results.py --file_path=PATH_TO_RESULTS --end_ind=21 --show_geom=True --show_geom_edge=True --show_stress=True --show_disp=False --geom_opacity=0.4 --stress_opacity=0.8 --disp_scale=100 --filename_prefix=eVTOL
```