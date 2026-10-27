#!/bin/sh
# Trust the egress proxy's CA so HTTPS through mitmproxy can be inspected, then start
# the tool server. The CA appears in the shared volume after the proxy's first start.
set -e
CA=/mitm/mitmproxy-ca-cert.pem
for i in $(seq 1 30); do [ -f "$CA" ] && break; sleep 1; done
if [ -f "$CA" ]; then
  cat /etc/ssl/certs/ca-certificates.crt "$CA" > /tmp/ca-bundle.pem
  export SSL_CERT_FILE=/tmp/ca-bundle.pem REQUESTS_CA_BUNDLE=/tmp/ca-bundle.pem CURL_CA_BUNDLE=/tmp/ca-bundle.pem
else
  echo "warning: proxy CA not found, HTTPS requests will fail certificate checks" >&2
fi
exec python /opt/tools/tool_server.py
