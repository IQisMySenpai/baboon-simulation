"""
We are going to use weak, non-interpreter-enforced protocols to define objects.
Further down the line we could eventually use
https://peps.python.org/pep-0544/, but objects right now are too simple to
require other machinery such as classes.
Refer to https://peps.python.org/pep-0020/: "Simple is better than complex."

Here is the documentation for said objects.

baboons: np.ndarray: (M, 2)-np.ndarray (i.e. Mx2 matrix)
    - M is the number of baboons.
    - 2 is the x and y coordinates of the baboon.

baboons_trajectory: (N, M, 2)-np.ndarray (i.e. NxMx2 matrix)
    - N is the number of steps.
    - M is the number of baboons.
    - 2 is the x and y coordinates of the baboon.
For example baboons_trajectory[t, i, :] is the position of baboon i at time t.

colors: List[str]: List of colors for each baboon. (List of length M)
"""
