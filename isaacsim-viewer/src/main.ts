/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

import './index.css';
import { AppStreamer, DirectConfig, StreamEvent, StreamProps, LogLevel, StreamType, eAction, eStatus } from '@nvidia/omniverse-webrtc-streaming-library';

interface AppState {
    streamFailed: boolean;
    errorMessage: string;
}

class StreamingApp {
    private streamConnected = false;
    private streamRequested = false;
    private state: AppState = {
        streamFailed: false,
        errorMessage: 'FAILED TO CONNECT TO STREAM',
    };

    private updateUI() {
        const mainDiv = document.getElementById('main-div');
        const errorMessage = document.getElementById('error-message');
        const loadingMessage = document.getElementById('loading-stream-message');
        if (!mainDiv || !errorMessage || !loadingMessage) return;
        const setDisplay = (el, val) => { if (el) el.style.display = val; };
        if (this.state.streamFailed) {
            setDisplay(mainDiv, 'none');
            setDisplay(errorMessage, 'flex');
            setDisplay(loadingMessage, 'none');
            const errorText = document.getElementById('error-message-text');
            if (errorText) errorText.textContent = this.state.errorMessage;
        } else if (this.streamConnected) {
            setDisplay(mainDiv, 'block');
            setDisplay(errorMessage, 'none');
            setDisplay(loadingMessage, 'none');
        } else {
            setDisplay(mainDiv, 'none');
            setDisplay(errorMessage, 'none');
            setDisplay(loadingMessage, 'flex');
        }
    }


    public async initialize() {
        // Check URL query parameter: ?server=<ip>
        const urlParams = new URLSearchParams(window.location.search);
        const serverFromUrl = urlParams.get('server');

        if (serverFromUrl) {
            // Auto-connect using URL parameter
            this.hideServerPanel();
            this.updateUI();
            await this.initializeStream(serverFromUrl);
        } else {
            // Show input panel and wait for user to click Connect
            this.setupServerPanel();
        }
    }

    private hideServerPanel() {
        const panel = document.getElementById('server-panel');
        if (panel) panel.classList.add('hidden');
    }

    private setupServerPanel() {
        const panel = document.getElementById('server-panel');
        const input = document.getElementById('server-input') as HTMLInputElement;
        const btn = document.getElementById('connect-btn');
        if (!panel || !input || !btn) return;

        // Pre-fill with last used IP from localStorage
        const lastIp = localStorage.getItem('isaacsim-server-ip');
        if (lastIp) input.value = lastIp;

        const doConnect = async () => {
            const ip = input.value.trim();
            if (!ip) { input.focus(); return; }
            localStorage.setItem('isaacsim-server-ip', ip);
            this.hideServerPanel();
            this.updateUI();
            await this.initializeStream(ip);
        };

        btn.addEventListener('click', doConnect);
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') doConnect(); });
        input.focus();
    }

    private async initializeStream(signalingServer: string) {
        if (this.streamConnected || this.streamRequested) {
            return;
        }
        this.streamRequested = true;
        console.info(`Connecting to signaling server: ${signalingServer}`);
        const streamConfig: DirectConfig = {
            videoElementId: 'remote-video',
            signalingServer,
            width: 1920,
            height: 1080,
            fps: 60,
            onStart: (message: StreamEvent) => {
                if (message.action === eAction.start) {
                    if (message.status === eStatus.success) {
                        this.streamConnected = true;
                        console.debug('Stream Ready');
                        this.updateUI();
                    }
                    else if (message.status === eStatus.error) {
                        console.error('Stream start error:', message.info);
                        this.state.streamFailed = true;
                        this.state.errorMessage = `${message.info || 'Unknown error'}`;
                        this.updateUI();
                    }
                }
            },
            onCustomEvent: (message: any) => {
                console.log('Custom event:', message);
            },
            onStop: (message: StreamEvent) => {
                this.streamConnected = false;
                console.log('Stream stopped:', message);
                this.updateUI();
            },
        };
        const streamProps: StreamProps = {
            streamSource: StreamType.DIRECT,
            logLevel: LogLevel.INFO,
            streamConfig,
        };
        try {
            const result = await AppStreamer.connect(streamProps);
            console.info(`Success: ${result.info}`);
        }
        catch (error) {
            console.error('Failed to connect to stream:', error);
            this.streamRequested = false;
            this.state.streamFailed = true;
            this.state.errorMessage = `${error || 'Unknown error'}`;
            this.updateUI();
        }
    }
}

// Initialize the application
const app = new StreamingApp();
void app.initialize();
