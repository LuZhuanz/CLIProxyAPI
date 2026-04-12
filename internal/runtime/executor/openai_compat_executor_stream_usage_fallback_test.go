package executor

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v6/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/executor"
	"github.com/router-for-me/CLIProxyAPI/v6/sdk/cliproxy/usage"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v6/sdk/translator"
)

type usageCollectPlugin struct {
	mu      sync.Mutex
	records []usage.Record
}

func (p *usageCollectPlugin) HandleUsage(_ context.Context, record usage.Record) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.records = append(p.records, record)
}

func (p *usageCollectPlugin) findRecord(provider, model string) (usage.Record, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := len(p.records) - 1; i >= 0; i-- {
		rec := p.records[i]
		if rec.Provider == provider && rec.Model == model {
			return rec, true
		}
	}
	return usage.Record{}, false
}

func TestOpenAICompatExecutorExecuteStream_UsageFallbackFromTranslatedCompleted(t *testing.T) {
	provider := "openai-compat-usage-fallback-test"
	model := "gpt-test"

	plugin := &usageCollectPlugin{}
	usage.RegisterPlugin(plugin)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"id\":\"chatcmpl_1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":8,\"output_tokens\":28,\"total_tokens\":36}}}\n\n"))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}))
	defer server.Close()

	executor := NewOpenAICompatExecutor(provider, &config.Config{})
	auth := &cliproxyauth.Auth{Attributes: map[string]string{"base_url": server.URL, "api_key": "test-key"}}
	payload := []byte(`{"model":"gpt-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`)

	result, err := executor.ExecuteStream(context.Background(), auth, cliproxyexecutor.Request{Model: model, Payload: payload}, cliproxyexecutor.Options{
		SourceFormat:    sdktranslator.FromString("openai"),
		OriginalRequest: payload,
		Stream:          true,
	})
	if err != nil {
		t.Fatalf("ExecuteStream error: %v", err)
	}

	for chunk := range result.Chunks {
		if chunk.Err != nil {
			t.Fatalf("stream chunk error: %v", chunk.Err)
		}
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		record, ok := plugin.findRecord(provider, model)
		if ok {
			if record.Detail.TotalTokens == 0 {
				t.Fatalf("expected non-zero usage tokens, got %#v", record.Detail)
			}
			if record.Detail.TotalTokens != 36 {
				t.Fatalf("unexpected total tokens: got %d want %d", record.Detail.TotalTokens, 36)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}

	t.Fatalf("usage record not published for provider=%s model=%s", provider, model)
}
