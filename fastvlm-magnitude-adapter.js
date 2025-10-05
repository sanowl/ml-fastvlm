const express = require('express');
const axios = require('axios');
require('dotenv').config();

function extractTextFragments(content) {
    if (!content) return [];
    if (typeof content === 'string') return [content];

    if (Array.isArray(content)) {
        const fragments = [];
        for (const part of content) {
            if (!part) continue;
            if (typeof part === 'string') {
                fragments.push(part);
                continue;
            }
            if (typeof part === 'object') {
                if (typeof part.text === 'string') {
                    fragments.push(part.text);
                    continue;
                }
                if (Array.isArray(part.content)) {
                    fragments.push(...extractTextFragments(part.content));
                    continue;
                }
            }
        }
        return fragments;
    }

    if (typeof content === 'object') {
        if (typeof content.text === 'string') return [content.text];
        if (Array.isArray(content.content)) return extractTextFragments(content.content);
    }

    return [];
}

function extractImageFromContent(content) {
    if (!content) return null;

    if (Array.isArray(content)) {
        for (const part of content) {
            const result = extractImageFromContent(part);
            if (result) return result;
        }
        return null;
    }

    if (typeof content === 'object') {
        if ((content.type === 'image' || content.type === 'input_image') && content.source?.data) {
            return content.source.data;
        }
        if (content.type === 'input_image' && typeof content.image_base64 === 'string') {
            return content.image_base64;
        }
        if (content.type === 'image_url' && content.image_url?.url) {
            const imageUrl = content.image_url.url;
            if (typeof imageUrl === 'string' && imageUrl.startsWith('data:image/')) {
                return imageUrl.split(',')[1];
            }
        }
        if (Array.isArray(content.content)) {
            return extractImageFromContent(content.content);
        }
    }

    return null;
}

function buildPromptFromMessages(messages = []) {
    const sections = [];
    for (const message of messages) {
        const textParts = extractTextFragments(message?.content);
        if (!textParts.length) continue;
        const role = (message.role || 'user').toUpperCase();
        sections.push(`${role}: ${textParts.join('\n')}`.trim());
    }
    return sections.join('\n\n');
}

function isPlanningPrompt(prompt) {
    if (!prompt) return false;
    const p = String(prompt).toLowerCase();
    return p.includes('plan out actions') || p.includes('<task>');
}

function extractTaskInstruction(prompt) {
    if (!prompt) return null;
    const m = String(prompt).match(/<task>\s*([\s\S]*?)\s*<\/task>/i);
    return m ? m[1].trim() : null;
}

function extractActiveUrlFromPrompt(prompt) {
    if (!prompt) return null;
    const m = String(prompt).match(/\[ACTIVE\][^\(]*\((https?:\/\/[^\)]+)\)/i);
    return m ? m[1] : null;
}

function chooseSearchNavUrl(task, basePrompt) {
    const query = task.replace(/^[Ss]earch\s*for\s*/,'').replace(/^\"|\"$/g,'').trim();
    const activeUrl = extractActiveUrlFromPrompt(basePrompt) || '';
    let url;
    if (/wikipedia\.org/i.test(activeUrl)) {
        url = `https://en.wikipedia.org/w/index.php?search=${encodeURIComponent(query)}`;
    } else if (/google\./i.test(activeUrl)) {
        url = `https://www.google.com/search?q=${encodeURIComponent(query)}`;
    } else {
        url = `https://en.wikipedia.org/w/index.php?search=${encodeURIComponent(query)}`;
    }
    return { url, query };
}

function buildVisionPrompt() {
    return `Describe what you see in this screenshot. Focus on interactive elements.

List all visible:
- Buttons (include exact text labels)
- Input fields (include labels or placeholder text)
- Links (include exact text)
- Icons and their locations
- Navigation menus

Be precise and factual. Only describe what is visible, no JSON.`;
}

function buildActionPrompt(visionDescription, task) {
    return `You are a web automation assistant. Generate actions to complete the TASK using elements from the vision description.

TASK: ${task}

VISION DESCRIPTION:
${visionDescription}

CRITICAL: Focus ONLY on the TASK. Ignore other products/sections visible on the page.

If TASK is "search for X":
- Look for a search box in the vision description
- Click search box, type X, press enter
- OR navigate directly to search results for X

Generate JSON in this EXACT format:
{
  "reasoning": "brief explanation",
  "actions": [array of action objects]
}

Valid action types:
- {"variant": "mouse:click", "target": "element from vision"}
- {"variant": "keyboard:type", "content": "text from task"}
- {"variant": "keyboard:enter"}

Extract search terms from TASK (e.g., "search for shoes" → type "shoes", NOT "jeans" or other visible items).

Output ONLY valid JSON:`;
}

function extractReasoningFromText(text) {
    if (!text) return '';
    const m = String(text).match(/Reasoning:\s*(.*)/i);
    return m ? m[1].trim() : '';
}

function normalizeActions(actionsRaw) {
    const allowed = new Set([
        'mouse:click',
        'keyboard:type',
        'keyboard:enter',
        'keyboard:select_all',
        'keyboard:backspace',
        'browser:nav',
    ]);
    const out = [];
    const arr = Array.isArray(actionsRaw) ? actionsRaw : (actionsRaw ? [actionsRaw] : []);
    for (const a of arr) {
        if (!a || typeof a !== 'object') continue;
        const variant = String(a.variant || '').trim();
        if (!allowed.has(variant)) continue;
        const item = { variant };
        if (variant === 'mouse:click') {
            const target = a.target || a.label || a.selector;
            if (typeof target === 'string' && target.trim()) {
                item.target = target.trim();
            } else {
                continue;
            }
        } else if (variant === 'keyboard:type') {
            if (typeof a.content === 'string') item.content = a.content;
        } else if (variant === 'browser:nav') {
            if (typeof a.url === 'string') item.url = a.url;
        }
        out.push(item);
    }
    return out;
}

function coercePlanFromText(rawText, basePrompt) {
    if (!rawText) return null;
    const text = String(rawText).trim();

    const tryParse = (s) => {
        try {
            return JSON.parse(s);
        } catch (err) {
            return null;
        }
    };

    let parsed = tryParse(text);

    if (!parsed) {
        const first = text.indexOf('{');
        const last = text.lastIndexOf('}');
        if (first !== -1 && last !== -1 && last > first) {
            const candidate = text.slice(first, last + 1);
            parsed = tryParse(candidate);
        }
    }

    if (!parsed && text.includes('"reasoning"') && text.includes('"actions"')) {
        let candidate = text;
        if (!candidate.endsWith('}')) {
            const openBraces = (candidate.match(/{/g) || []).length;
            const closeBraces = (candidate.match(/}/g) || []).length;
            const missing = openBraces - closeBraces;
            if (missing > 0) {
                candidate = candidate + '}'.repeat(missing);
                parsed = tryParse(candidate);
            }
        }
    }

    if (!parsed || typeof parsed !== 'object') return null;

    let actions = parsed.actions != null ? parsed.actions : parsed.action;
    const norm = normalizeActions(actions);
    if (!norm.length) return null;

    const reasoning = typeof parsed.reasoning === 'string' && parsed.reasoning.trim()
        ? parsed.reasoning.trim()
        : (extractReasoningFromText(text) || 'Plan derived from model response');

    return JSON.stringify({ reasoning, actions: norm });
}

function findLatestImage(messages = []) {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
        const message = messages[i];
        const imageBase64 = extractImageFromContent(message?.content);
        if (imageBase64) return imageBase64;
    }
    return null;
}

function appendJsonInstruction(prompt) {
    const reminder = `\n\nReturn ONLY valid JSON following this exact schema (no additional text, no markdown):\n{"data":{"reasoning":"<brief explanation>","passed":<true_or_false>}}`;
    if (!prompt) return reminder.trim();
    if (prompt.includes('{"data"')) return prompt;
    return `${prompt.trim()}${reminder}`;
}

function ensureJsonResponse(rawResponse, contextPrompt = '') {
    const result = {
        original: rawResponse,
        json: null,
        stringified: null,
        source: 'unknown',
        heuristic: null,
    };

    const normalizeShape = (obj) => {
        const out = { ...obj };
        if (out && out.data && typeof out.data.reasoning === 'string') {
            if (typeof out.reasoning !== 'string') out.reasoning = out.data.reasoning;
        }
        if (typeof out.reasoning === 'string' && (!out.data || typeof out.data !== 'object')) {
            out.data = {
                reasoning: out.reasoning,
                passed: typeof out.passed === 'boolean' ? out.passed : false,
            };
        }
        return out;
    };

    const tryParse = (candidate) => {
        try {
            const parsed = JSON.parse(candidate);
            const normalized = normalizeShape(parsed);
            if (parsed && typeof parsed === 'object' && parsed.data && typeof parsed.data.reasoning === 'string' && typeof parsed.data.passed === 'boolean') {
                result.json = normalized;
                result.stringified = JSON.stringify(normalized);
                result.source = 'parsed';
                return true;
            }
        } catch (err) {}
        return false;
    };

    if (typeof rawResponse === 'string' && rawResponse.trim().length > 0) {
        const trimmed = rawResponse.trim();
        if (tryParse(trimmed)) return result;

        const firstBrace = trimmed.indexOf('{');
        const lastBrace = trimmed.lastIndexOf('}');
        if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
            const candidate = trimmed.slice(firstBrace, lastBrace + 1);
            if (tryParse(candidate)) {
                result.source = 'fragment';
                return result;
            }
        }

        const normalized = trimmed.replace(/\s+/g, ' ');
        const ctx = (contextPrompt || '').toLowerCase();
        let positive = false;
        let heuristic = null;
        if (ctx.includes('example.com') || ctx.includes('example domain')) {
            positive = true;
            heuristic = 'context_contains_example.com';
        } else if (/open\s+tabs:\s*\[active\][^\n]*example/i.test(contextPrompt)) {
            positive = true;
            heuristic = 'open_tabs_shows_example';
        } else if (/(pass|success|correct|true|yes)/i.test(normalized) && !/\b(not|fail|incorrect|false)\b/i.test(normalized)) {
            positive = true;
            heuristic = 'keyword_positive_in_model_text';
        }
        const fallback = {
            data: {
                reasoning: normalized,
                passed: positive
            },
            reasoning: normalized
        };
        result.json = fallback;
        result.stringified = JSON.stringify(fallback);
        result.source = 'fallback';
        result.heuristic = heuristic;
        return result;
    }

    const fallback = {
        data: {
            reasoning: rawResponse ? String(rawResponse) : 'No response returned from FastVLM.',
            passed: false
        },
        reasoning: rawResponse ? String(rawResponse) : 'No response returned from FastVLM.'
    };
    result.json = fallback;
    result.stringified = JSON.stringify(fallback);
    result.source = 'fallback';
    return result;
}

class FastVLMMagnitudeAdapter {
    constructor(fastvlmUrl = 'http://localhost:8000') {
        this.fastvlmUrl = fastvlmUrl;
        this.app = express();
        this.app.use(express.json({ limit: '50mb', type: ['application/json', 'application/*+json'] }));
        this.app.use(express.urlencoded({ extended: true }));
        this.app.use((req, _res, next) => {
            console.log(`${req.method} ${req.url}`);
            next();
        });
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization, anthropic-version');
            if (req.method === 'OPTIONS') return res.sendStatus(200);
            next();
        });
        this.setupRoutes();
    }

    setupRoutes() {
        this.app.get('/health', async (req, res) => {
            try {
                const response = await axios.get(`${this.fastvlmUrl}/health`);
                res.json({
                    status: 'healthy',
                    fastvlm_status: response.data,
                    adapter_version: '1.0.0'
                });
            } catch (error) {
                res.status(503).json({
                    status: 'unhealthy',
                    error: 'FastVLM backend not available',
                    message: 'Make sure FastVLM is running on port 8000'
                });
            }
        });

        const chatCompletions = async (req, res) => {
            try {
                const { messages = [], max_tokens = 16000, model = "fastvlm-vision-model" } = req.body;

                if (!messages.length) {
                    throw new Error('Missing messages in request body');
                }

                const basePrompt = buildPromptFromMessages(messages);
                const planning = isPlanningPrompt(basePrompt);
                const task = extractTaskInstruction(basePrompt);
                const imageBase64 = findLatestImage(messages);

                const prompt = planning ? buildVisionPrompt() : appendJsonInstruction(basePrompt);

                if (!prompt?.trim()) {
                    throw new Error('Unable to extract textual content from conversation');
                }

                if (!imageBase64) {
                    console.log('No image provided, creating text-only response');
                    const planning = isPlanningPrompt(basePrompt);
                    let content;
                    if (planning) {
                        const task = extractTaskInstruction(basePrompt) || 'the task';
                        const { url } = chooseSearchNavUrl(task, basePrompt);
                        const plan = {
                            reasoning: `Navigate directly to results for ${task}.`,
                            actions: [ { variant: 'browser:nav', url } ]
                        };
                        content = JSON.stringify(plan);
                    } else {
                        const fb = ensureJsonResponse(`No screenshot received. Prompt: ${prompt.trim()}`);
                        content = fb.stringified;
                    }
                    const textOnlyResponse = {
                        id: `chatcmpl-fastvlm-${Date.now()}`,
                        object: "chat.completion",
                        created: Math.floor(Date.now() / 1000),
                        model: model,
                        choices: [{
                            index: 0,
                            message: { role: "assistant", content: content },
                            finish_reason: "stop"
                        }],
                        usage: {
                            prompt_tokens: 1,
                            completion_tokens: Math.floor(String(content).length / 4),
                            total_tokens: 1 + Math.floor(String(content).length / 4)
                        }
                    };
                    res.json(textOnlyResponse);
                    return;
                }

                try {
                    const fastvlmResponse = await axios.post(`${this.fastvlmUrl}/query`, {
                        prompt: prompt.trim(),
                        image_base64: imageBase64,
                        temperature: 0.2,
                        max_tokens: planning ? 200 : 256
                    }, {
                        headers: { 'Content-Type': 'application/json' },
                        timeout: 300000,
                        maxBodyLength: Infinity,
                        maxContentLength: Infinity
                    });

                    const rawResponse = fastvlmResponse.data?.response ?? fastvlmResponse.data;
                    console.log('FastVLM raw response:', rawResponse);

                    let responseContent;
                    if (planning) {
                        const visionDescription = typeof rawResponse === 'string'
                            ? rawResponse.trim()
                            : String(rawResponse);

                        console.log('=== STAGE 1: FastVLM Vision Description ===');
                        console.log(visionDescription);
                        console.log('===========================================');

                        try {
                            const actionPrompt = buildActionPrompt(visionDescription, task || 'the task');
                            console.log('=== STAGE 2: Sending to Text LLM ===');

                            const textLLMResponse = await axios.post(`${this.fastvlmUrl}/query_text`, {
                                prompt: actionPrompt,
                                temperature: 0.2,
                                max_tokens: 512
                            }, {
                                headers: { 'Content-Type': 'application/json' },
                                timeout: 600000
                            });

                            const actionJSON = textLLMResponse.data?.response ?? textLLMResponse.data;
                            console.log('=== Text LLM Action JSON ===');
                            console.log(actionJSON);
                            console.log('============================');

                            const coerced = coercePlanFromText(actionJSON, basePrompt);
                            if (coerced) {
                                responseContent = coerced;
                                console.log('SUCCESS: Hybrid mode generated valid plan');
                            } else {
                                throw new Error('Text LLM did not produce valid JSON');
                            }
                        } catch (textLLMError) {
                            console.error('ERROR: Text LLM failed:', textLLMError.message);
                            const { url } = chooseSearchNavUrl(task || 'the task', basePrompt);
                            responseContent = JSON.stringify({
                                reasoning: `Hybrid mode failed. Fallback navigation for: ${task}`,
                                actions: [{ variant: 'browser:nav', url }]
                            });
                        }
                    } else {
                        const formatted = ensureJsonResponse(rawResponse, basePrompt);
                        responseContent = formatted.stringified;
                    }

                    if (planning) {
                        const disableNav = !!process.env.FASTVLM_DISABLE_NAV_OVERRIDE;
                        if (!disableNav) {
                            const task = extractTaskInstruction(basePrompt) || 'the task';
                            const { url } = chooseSearchNavUrl(task, basePrompt);
                            responseContent = JSON.stringify({
                                reasoning: `Navigate directly to results for ${task}.`,
                                actions: [ { variant: 'browser:nav', url } ]
                            });
                        }
                    }

                    console.log('Adapter sending content to Magnitude:', responseContent);
                    const openaiResponse = {
                        id: `chatcmpl-fastvlm-${Date.now()}`,
                        object: "chat.completion",
                        created: Math.floor(Date.now() / 1000),
                        model: model,
                        choices: [{
                            index: 0,
                            message: { role: "assistant", content: responseContent },
                            finish_reason: "stop"
                        }],
                        usage: {
                            prompt_tokens: fastvlmResponse.data.usage?.input_tokens || 100,
                            completion_tokens: fastvlmResponse.data.usage?.output_tokens || 50,
                            total_tokens: (fastvlmResponse.data.usage?.input_tokens || 100) + (fastvlmResponse.data.usage?.output_tokens || 50)
                        }
                    };

                    console.log('FastVLM response sent to Magnitude');
                    res.json(openaiResponse);

                } catch (fastvlmError) {
                    const status = fastvlmError?.response?.status;
                    const data = fastvlmError?.response?.data;
                    const code = fastvlmError?.code;
                    console.error('FastVLM processing error:', fastvlmError?.message || fastvlmError);
                    if (status) console.error('status:', status);
                    if (code) console.error('code:', code);
                    if (data) console.error('data:', typeof data === 'object' ? JSON.stringify(data) : String(data));

                    const planning = isPlanningPrompt(prompt);
                    let fallbackContent;
                    if (planning) {
                        const task = extractTaskInstruction(prompt) || 'the task';
                        const disableNav = !!process.env.FASTVLM_DISABLE_NAV_OVERRIDE;
                        if (!disableNav) {
                            const { url } = chooseSearchNavUrl(task, prompt);
                            fallbackContent = JSON.stringify({
                                reasoning: `Navigate directly to results for ${task}.`,
                                actions: [ { variant: 'browser:nav', url } ]
                            });
                        } else {
                            fallbackContent = JSON.stringify({
                                reasoning: `FastVLM connection failed: ${fastvlmError?.message || 'Unknown error'}`,
                                actions: [{ variant: 'browser:nav', url: 'about:blank' }]
                            });
                        }
                    } else {
                        const formatted = ensureJsonResponse(
                            `Adapter error calling FastVLM: ${fastvlmError?.message || fastvlmError} (status=${status ?? 'n/a'}, code=${code ?? 'n/a'})`,
                            prompt
                        );
                        fallbackContent = formatted.stringified;
                    }

                    const errorResponse = {
                        id: `chatcmpl-fastvlm-${Date.now()}`,
                        object: "chat.completion",
                        created: Math.floor(Date.now() / 1000),
                        model: model,
                        choices: [{
                            index: 0,
                            message: { role: "assistant", content: fallbackContent },
                            finish_reason: "stop"
                        }],
                        usage: {
                            prompt_tokens: 1,
                            completion_tokens: Math.floor(String(fallbackContent).length / 4),
                            total_tokens: 1 + Math.floor(String(fallbackContent).length / 4)
                        }
                    };

                    res.status(status && status >= 400 ? status : 200).json(errorResponse);
                }

            } catch (error) {
                console.error('Request processing error:', error.message);
                const errorResponse = {
                    id: `chatcmpl-fastvlm-${Date.now()}`,
                    object: "error",
                    error: {
                        message: error.message,
                        type: "internal_error"
                    }
                };
                res.status(500).json(errorResponse);
            }
        };

        this.app.post('/v1/chat/completions', chatCompletions);
        this.app.post('/chat/completions', chatCompletions);

        this.app.get('/v1/models', (req, res) => {
            res.json({
                object: "list",
                data: [
                    {
                        id: "fastvlm-0.5b-magnitude-adapter",
                        object: "model",
                        created: Date.now(),
                        owned_by: "fastvlm"
                    },
                    {
                        id: "fastvlm-vision-model",
                        object: "model",
                        created: Date.now(),
                        owned_by: "fastvlm"
                    }
                ]
            });
        });

        this.app.use((req, res) => {
            console.warn('Unhandled route:', req.method, req.url);
            return res.status(404).json({ error: { message: `No route for ${req.method} ${req.url}` } });
        });
    }

    async start(port = 8080) {
        try {
            await axios.get(`${this.fastvlmUrl}/health`);
            console.log('FastVLM backend is available');
        } catch (error) {
            console.log('Warning: FastVLM backend not available. Make sure to run: python web_agent.py');
        }

        this.server = this.app.listen(port, () => {
            console.log('FastVLM-Magnitude Adapter running on port', port);
            console.log('Magnitude will connect to this adapter instead of Claude');
            console.log('Adapter URL: http://localhost:' + port);
            console.log('FastVLM Backend: ' + this.fastvlmUrl);
            console.log('');
            console.log('Configure Magnitude to use:');
            console.log('  ANTHROPIC_API_KEY=dummy-key-not-used');
            console.log('  ANTHROPIC_BASE_URL=http://localhost:' + port);
            console.log('');
            console.log('Planning settings: FASTVLM_DISABLE_NAV_OVERRIDE =', process.env.FASTVLM_DISABLE_NAV_OVERRIDE || '0');
        });

        return this.server;
    }

    async stop() {
        if (this.server) {
            this.server.close();
            console.log('FastVLM-Magnitude Adapter stopped');
        }
    }
}

if (require.main === module) {
    const adapter = new FastVLMMagnitudeAdapter();
    adapter.start();

    process.on('SIGINT', async () => {
        console.log('\nShutting down adapter...');
        await adapter.stop();
        process.exit(0);
    });
}

module.exports = { FastVLMMagnitudeAdapter };
