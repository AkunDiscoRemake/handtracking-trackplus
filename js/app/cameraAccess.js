// Camera access: environment detection, permission state, the request
// itself (with constraint fallbacks) and a classification of every failure
// into something the user can act on.

const UA = navigator.userAgent || '';

export function detectEnvironment() {
  const android = /Android/i.test(UA);
  const ios = /iPhone|iPad|iPod/i.test(UA) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  // In-app browsers (embedded WebViews) usually refuse getUserMedia without
  // ever showing a prompt.
  const inAppName =
    /Instagram/i.test(UA) ? 'Instagram'
      : /FBAN|FBAV|FB_IAB/i.test(UA) ? 'Facebook'
        : /WhatsApp/i.test(UA) ? 'WhatsApp'
          : /Line\//i.test(UA) ? 'LINE'
            : /musical_ly|BytedanceWebview|TikTok/i.test(UA) ? 'TikTok'
              : /Snapchat/i.test(UA) ? 'Snapchat'
                : /MicroMessenger/i.test(UA) ? 'WeChat'
                  : /Twitter/i.test(UA) ? 'X/Twitter'
                    : android && /; wv\)/.test(UA) ? 'um app'
                      : ios && !/Safari\//.test(UA) && /AppleWebKit/.test(UA) ? 'um app'
                        : null;
  return {
    android,
    ios,
    inApp: !!inAppName,
    inAppName,
    secure: window.isSecureContext,
    hasApi: !!navigator.mediaDevices?.getUserMedia,
  };
}

// 'granted' | 'denied' | 'prompt' | 'unknown' (Permissions API missing, e.g. older Safari)
export async function cameraPermission() {
  try {
    const st = await navigator.permissions.query({ name: 'camera' });
    return st;
  } catch {
    return null;
  }
}

// Opens the camera. Tries the requested facing mode first, then any camera,
// so a phone without a matching lens still works.
export async function openCamera(facingMode) {
  const env = detectEnvironment();
  if (!env.secure) throw classify({ name: 'InsecureContext' });
  if (!env.hasApi) throw classify({ name: env.inApp ? 'InAppBrowser' : 'NoApi' });
  const base = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 60 } };
  const t0 = performance.now();
  try {
    return await navigator.mediaDevices.getUserMedia({ audio: false, video: { ...base, facingMode: { ideal: facingMode } } });
  } catch (e) {
    if (e?.name === 'OverconstrainedError' || e?.name === 'NotFoundError') {
      try { return await navigator.mediaDevices.getUserMedia({ audio: false, video: true }); } catch (e2) { throw classify(e2, performance.now() - t0); }
    }
    throw classify(e, performance.now() - t0);
  }
}

// Maps a DOMException to { kind, error }.
export function classify(e, elapsedMs = 0) {
  const env = detectEnvironment();
  const n = e?.name || '';
  let kind = 'unknown';
  if (n === 'InsecureContext') kind = 'insecure';
  else if (n === 'NoApi') kind = 'noapi';
  else if (n === 'InAppBrowser') kind = 'inapp';
  else if (n === 'NotAllowedError' || n === 'PermissionDeniedError' || n === 'SecurityError') {
    // A refusal that arrives instantly without a prompt inside an embedded
    // browser is the WebView policy, not the user.
    kind = env.inApp ? 'inapp' : elapsedMs < 250 ? 'blocked' : 'denied';
  } else if (n === 'NotFoundError' || n === 'DevicesNotFoundError' || n === 'OverconstrainedError') kind = 'nocamera';
  else if (n === 'NotReadableError' || n === 'TrackStartError' || n === 'AbortError') kind = 'busy';
  // Some browsers/WebViews refuse with NotSupportedError instead of
  // NotAllowedError when camera access is disabled for the page.
  else if (n === 'NotSupportedError') kind = 'blocked';
  // Inside an embedded browser any refusal is the WebView's, whatever the name.
  if (env.inApp && ['unknown', 'denied', 'blocked', 'noapi'].includes(kind)) kind = 'inapp';
  const err = new Error(e?.message || n || 'camera error');
  err.kind = kind;
  err.original = e;
  return err;
}

export function chromeIntentUrl() {
  const { host, pathname, search } = location;
  return `intent://${host}${pathname}${search}#Intent;scheme=https;package=com.android.chrome;S.browser_fallback_url=${encodeURIComponent(location.href)};end`;
}

// Human instructions per failure kind and platform.
export function helpFor(kind, env = detectEnvironment()) {
  const resetAndroid = [
    'Toque no ícone à esquerda do endereço (cadeado ou ⓘ).',
    'Abra “Permissões” e mude “Câmera” para “Permitir”.',
    'Toque em “Tentar de novo” aqui embaixo.',
  ];
  const resetIos = [
    'Toque em “aA” na barra de endereço do Safari.',
    'Escolha “Ajustes do Site” e mude “Câmera” para “Permitir”.',
    'Toque em “Tentar de novo” aqui embaixo.',
  ];
  const resetDesktop = [
    'Clique no ícone à esquerda do endereço (cadeado ou ajustes).',
    'Mude “Câmera” para “Permitir” e recarregue a página.',
  ];
  switch (kind) {
    case 'inapp':
      return {
        title: `O navegador interno de ${env.inAppName || 'um app'} não libera a câmera`,
        steps: env.ios
          ? ['Toque no menu (•••) ou no ícone de compartilhar.', 'Escolha “Abrir no Safari”.', 'No Safari, toque em “Permitir câmera”.']
          : ['Abra esta página no Chrome com o botão abaixo.', 'No Chrome, toque em “Permitir câmera” e aceite o pedido.'],
        actions: env.android ? ['openChrome', 'copy'] : ['copy'],
      };
    case 'denied':
    case 'blocked':
      return {
        title: kind === 'blocked' ? 'O navegador bloqueou a câmera sem perguntar' : 'A permissão da câmera foi negada',
        steps: [
          ...(env.android ? resetAndroid : env.ios ? resetIos : resetDesktop),
          ...(env.android ? ['Se continuar bloqueado: Configurações do Android → Apps → Chrome → Permissões → Câmera → Permitir.'] : []),
          ...(env.ios ? ['Se continuar bloqueado: Ajustes do iPhone → Safari → Câmera → Perguntar ou Permitir.'] : []),
        ],
        actions: ['retry', 'reload'],
      };
    case 'busy':
      return {
        title: 'A câmera está em uso por outro app',
        steps: ['Feche apps que usam a câmera (chamadas de vídeo, câmera, outras abas).', 'Toque em “Tentar de novo”.'],
        actions: ['retry'],
      };
    case 'nocamera':
      return { title: 'Nenhuma câmera encontrada', steps: ['Conecte ou habilite uma câmera e tente de novo.'], actions: ['retry'] };
    case 'insecure':
      return { title: 'A câmera só funciona em endereço seguro (https)', steps: [`Abra ${location.href.replace(/^http:/, 'https:')}`], actions: ['copy'] };
    case 'noapi':
      return { title: 'Este navegador não oferece acesso à câmera', steps: ['Use o Chrome (Android) ou o Safari (iPhone) atualizados.'], actions: ['copy'] };
    default:
      return { title: 'Não foi possível abrir a câmera', steps: ['Toque em “Tentar de novo”. Se persistir, recarregue a página.'], actions: ['retry', 'reload'] };
  }
}
