#!/bin/bash
set -euo pipefail

mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml <<'EOF'
[server]
port = 8888
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
EOF

pip install --user -r protein-property-predictor/env/requirements.txt || true
cd protein-property-predictor
exec streamlit run app.py --server.port=8888 --server.address=0.0.0.0
