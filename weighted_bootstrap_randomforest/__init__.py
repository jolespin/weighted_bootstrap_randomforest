# ==============
# weighted_bootstrap_randomforest
# ==============
# Implementations of Weighted bootstrap Random Forests classifier and regressors
# ------------------------------------
# GitHub: https://github.com/jolespin/weighted_bootstrap_randomforest
# PyPI: https://pypi.org/project/weighted_bootstrap_randomforest/
# ------------------------------------
# =======
# Contact
# =======
# Producer: Josh L. Espinoza
# Contact: jespinoz@jcvi.org, jol.espinoz@gmail.com
# Google Scholar: https://scholar.google.com/citations?user=r9y1tTQAAAAJ&hl
# =======
# License BSD-3
# =======
# https://opensource.org/licenses/BSD-3-Clause
#
# Copyright 2018 Josh L. Espinoza
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# =======
# Version
# =======
__version__= "2022.9.26"
__author__ = "Josh L. Espinoza"
__email__ = "jespinoz@jcvi.org, jol.espinoz@gmail.com"
__url__ = "https://github.com/jolespin/weighted_bootstrap_randomforest"
__license__ = "BSD-3"
__developmental__ = True

# =======
# Direct Exports
# =======
__functions__ = [
"_check_pandas",
"_generate_sample_indices_weighted_bootstrap",
"_parallel_build_trees_weighted_bootstrap",


]
__classes__ = [
   "WeightedBootstrapRandomForestClassifier",
   "WeightedBootstrapRandomForestRegressor",
]

__all__ = sorted(__functions__ + __classes__)

from .weighted_bootstrap_randomforest import *
