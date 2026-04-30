/* ---------- SVG placeholders ---------- */
function svgPlaceholderSimple(){
  const svg =
`<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f1f5f9"/>
      <stop offset="1" stop-color="#e2e8f0"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="600" height="600" fill="url(#g)"/>
  <rect x="18" y="18" width="564" height="564" rx="26" ry="26" fill="rgba(0,0,0,0.03)" stroke="rgba(0,0,0,0.10)"/>
  <g transform="translate(170,190)" fill="rgba(0,0,0,0.18)">
    <rect x="0" y="0" width="260" height="210" rx="18" ry="18" fill="rgba(0,0,0,0.06)" stroke="rgba(0,0,0,0.16)" />
    <circle cx="72" cy="70" r="18" />
    <path d="M30 180 L110 115 L160 155 L205 125 L235 180 Z" />
  </g>
</svg>`;
  return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svg)));
}

function svgPlaceholderNotFound(text){
  const t = escapeHtml(text || "Dataset not found");
  const svg =
`<svg xmlns="http://www.w3.org/2000/svg" width="600" height="600">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#f8fafc"/>
      <stop offset="1" stop-color="#eef2ff"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="600" height="600" fill="url(#g)"/>
  <rect x="18" y="18" width="564" height="564" rx="26" ry="26" fill="rgba(0,0,0,0.02)" stroke="rgba(0,0,0,0.10)"/>
  <g transform="translate(0,0)">
    <text x="300" y="290" text-anchor="middle" font-size="30" font-family="ui-sans-serif,system-ui" fill="rgba(0,0,0,0.62)">${t}</text>
    <text x="300" y="335" text-anchor="middle" font-size="18" font-family="ui-sans-serif,system-ui" fill="rgba(0,0,0,0.45)">folder missing or moved</text>
  </g>
</svg>`;
  return "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(svg)));
}
