#!/bin/sh
# Copy the submission, reference, checks and harness into ~/w on the sandbox.
set -e
T=/private/tmp/claude-501/-Users-shivam-temp-muse-challenge/161d0400-c44b-492f-aef0-c49601f8eb92/scratchpad
B64=$(COPYFILE_DISABLE=1 tar --no-xattrs -C /Users/shivam/temp/muse-challenge -cf - submission.py reference.py checks.py | cat - | base64)
H64=$(COPYFILE_DISABLE=1 tar --no-xattrs -C $T/test -cf - bench.py | base64)
daytona exec okc-attn-h100 --timeout 60 -- bash -lc "mkdir -p ~/w && cd ~/w && echo $B64 | base64 -d | tar -xf - && echo $H64 | base64 -d | tar -xf - && ls" 2>&1 | grep -v level=warning
