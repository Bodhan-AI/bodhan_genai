# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Test-suite defaults.

``serving.service`` installs Basic auth at import time and refuses to be
imported without an auth decision -- deliberately, because forgetting to export
the credential used to serve every endpoint openly. The test suite is not a
server, so it opts out here, in one obvious place, rather than each test
arranging it.
"""

import os

os.environ.setdefault("ASR_AUTH_ALLOW_OPEN", "1")
