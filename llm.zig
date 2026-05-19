const std = @import("std");
const root = @import("main.zig");

pub const SseWriter = struct {
    conn: *std.net.Server.Connection,
    allocator: std.mem.Allocator,

    pub fn write(self: *SseWriter, data: []const u8) !void {
        const line = try std.fmt.allocPrint(self.allocator, "data: {s}\n\n", .{data});
        defer self.allocator.free(line);
        try self.conn.stream.writeAll(line);
    }

    pub fn ping(self: *SseWriter) !void {
        try self.conn.stream.writeAll(": ping\n\n");
    }
};

pub fn writeJsonEvent(allocator: std.mem.Allocator, sse: *SseWriter, fields: anytype) !void {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const writer = buf.writer();
    try writer.writeByte('{');
    const fields_info = @typeInfo(@TypeOf(fields)).@"struct";
    inline for (fields_info.fields, 0..) |field, i| {
        if (i > 0) try writer.writeByte(',');
        try root.writeJsonString(writer, field.name);
        try writer.writeByte(':');
        const val = @field(fields, field.name);
        const ValType = @TypeOf(val);
        switch (@typeInfo(ValType)) {
            .pointer => |ptr| {
                if (ptr.child == u8) {
                    try root.writeJsonString(writer, val);
                } else {
                    try writer.writeAll("null");
                }
            },
            .int, .comptime_int => {
                try std.fmt.format(writer, "{d}", .{val});
            },
            .bool => {
                try writer.writeAll(if (val) "true" else "false");
            },
            else => {
                try writer.writeAll("null");
            },
        }
    }
    try writer.writeByte('}');
    try sse.write(buf.items);
}

pub const OpenAIStreamChunk = struct {
    content: ?[]u8,
    reasoning_content: ?[]u8,
    finish_reason: ?[]u8,
    tool_calls: ?std.ArrayList(OpenAIToolCallDelta),
    allocator: std.mem.Allocator,

    pub fn deinit(self: *OpenAIStreamChunk) void {
        if (self.content) |c| self.allocator.free(c);
        if (self.reasoning_content) |r| self.allocator.free(r);
        if (self.finish_reason) |f| self.allocator.free(f);
        if (self.tool_calls) |*tcs| {
            for (tcs.items) |*tc| tc.deinit(self.allocator);
            tcs.deinit();
        }
    }
};

pub const OpenAIToolCallDelta = struct {
    index: ?usize,
    id: ?[]u8,
    name: ?[]u8,
    arguments: ?[]u8,

    pub fn deinit(self: *OpenAIToolCallDelta, allocator: std.mem.Allocator) void {
        if (self.id) |id| allocator.free(id);
        if (self.name) |n| allocator.free(n);
        if (self.arguments) |a| allocator.free(a);
    }
};

pub fn parseSseLine(allocator: std.mem.Allocator, line: []const u8) !?OpenAIStreamChunk {
    if (!std.mem.startsWith(u8, line, "data: ")) return null;
    const data = line[6..];
    if (std.mem.eql(u8, data, "[DONE]")) return null;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, data, .{}) catch return null;
    defer parsed.deinit();

    const choices = parsed.value.object.get("choices") orelse return null;
    if (choices != .array or choices.array.items.len == 0) return null;

    const choice = choices.array.items[0];
    if (choice != .object) return null;

    const delta = choice.object.get("delta") orelse return null;
    if (delta != .object) return null;

    var chunk = OpenAIStreamChunk{
        .content = null,
        .reasoning_content = null,
        .finish_reason = null,
        .tool_calls = null,
        .allocator = allocator,
    };

    if (delta.object.get("content")) |c| {
        if (c == .string and c.string.len > 0) {
            chunk.content = try allocator.dupe(u8, c.string);
        }
    }
    if (delta.object.get("reasoning_content")) |r| {
        if (r == .string and r.string.len > 0) {
            chunk.reasoning_content = try allocator.dupe(u8, r.string);
        }
    }

    if (choice.object.get("finish_reason")) |fr| {
        if (fr == .string) {
            chunk.finish_reason = try allocator.dupe(u8, fr.string);
        }
    }

    if (delta.object.get("tool_calls")) |tcs_val| {
        if (tcs_val == .array) {
            chunk.tool_calls = std.ArrayList(OpenAIToolCallDelta).init(allocator);
            for (tcs_val.array.items) |tc_val| {
                if (tc_val != .object) continue;
                var tc_delta = OpenAIToolCallDelta{
                    .index = null,
                    .id = null,
                    .name = null,
                    .arguments = null,
                };
                if (tc_val.object.get("index")) |idx| {
                    if (idx == .integer) tc_delta.index = @intCast(idx.integer);
                }
                if (tc_val.object.get("id")) |id| {
                    if (id == .string and id.string.len > 0) {
                        tc_delta.id = try allocator.dupe(u8, id.string);
                    }
                }
                if (tc_val.object.get("function")) |func| {
                    if (func == .object) {
                        if (func.object.get("name")) |n| {
                            if (n == .string and n.string.len > 0) {
                                tc_delta.name = try allocator.dupe(u8, n.string);
                            }
                        }
                        if (func.object.get("arguments")) |a| {
                            if (a == .string and a.string.len > 0) {
                                tc_delta.arguments = try allocator.dupe(u8, a.string);
                            }
                        }
                    }
                }
                try chunk.tool_calls.?.append(tc_delta);
            }
        }
    }

    return chunk;
}

pub const ToolCallAccumulator = struct {
    id: []u8,
    name: []u8,
    arg_parts: std.ArrayList([]u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !*ToolCallAccumulator {
        const self = try allocator.create(ToolCallAccumulator);
        self.* = .{
            .id = try allocator.dupe(u8, ""),
            .name = try allocator.dupe(u8, ""),
            .arg_parts = std.ArrayList([]u8).init(allocator),
            .allocator = allocator,
        };
        return self;
    }

    pub fn deinit(self: *ToolCallAccumulator) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
        for (self.arg_parts.items) |part| self.allocator.free(part);
        self.arg_parts.deinit();
        self.allocator.destroy(self);
    }

    pub fn getArguments(self: *ToolCallAccumulator) ![]u8 {
        var total: usize = 0;
        for (self.arg_parts.items) |part| total += part.len;
        const result = try self.allocator.alloc(u8, total);
        var offset: usize = 0;
        for (self.arg_parts.items) |part| {
            @memcpy(result[offset .. offset + part.len], part);
            offset += part.len;
        }
        return result;
    }
};

pub fn callOpenAIStreamingOnce(allocator: std.mem.Allocator, base_url: []const u8, api_key: []const u8, body: []const u8, callback: anytype, ctx: anytype, started: *bool) !void {
    const url_str = try std.fmt.allocPrint(allocator, "{s}/chat/completions", .{base_url});
    defer allocator.free(url_str);

    const auth_header = try std.fmt.allocPrint(allocator, "Authorization: Bearer {s}", .{api_key});
    defer allocator.free(auth_header);

    var child = std.process.Child.init(&[_][]const u8{
        "curl",
        "-sS",
        "-N",
        "--no-buffer",
        "--fail-with-body",
        "--max-time", "600",
        "--connect-timeout", "30",
        "--retry", "0",
        "-X", "POST",
        url_str,
        "-H", auth_header,
        "-H", "Content-Type: application/json",
        "-H", "Accept: text/event-stream",
        "--data-binary", "@-",
    }, allocator);
    child.stdin_behavior = .Pipe;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    if (child.stdin) |stdin| {
        stdin.writeAll(body) catch {};
        stdin.close();
        child.stdin = null;
    }

    var stderr_buf = std.ArrayList(u8).init(allocator);
    defer stderr_buf.deinit();

    var line_buf = std.ArrayList(u8).init(allocator);
    defer line_buf.deinit();

    var read_err: ?anyerror = null;
    if (child.stdout) |stdout| {
        var reader = stdout.reader();
        while (true) {
            const byte = reader.readByte() catch |err| {
                if (err != error.EndOfStream) read_err = err;
                break;
            };
            if (byte == '\n') {
                const line = std.mem.trimRight(u8, line_buf.items, "\r");
                if (line.len > 0) {
                    started.* = true;
                    callback(allocator, line, ctx) catch |err| {
                        read_err = err;
                        break;
                    };
                }
                line_buf.clearRetainingCapacity();
            } else {
                try line_buf.append(byte);
            }
        }
    }

    if (child.stderr) |stderr| {
        stderr.reader().readAllArrayList(&stderr_buf, 65536) catch {};
    }

    const term = child.wait() catch |err| {
        std.log.err("curl wait failed: {}", .{err});
        return err;
    };

    switch (term) {
        .Exited => |code| {
            if (code != 0) {
                std.log.err("curl exit code {d}: {s}", .{ code, stderr_buf.items });
                return error.CurlFailed;
            }
        },
        else => {
            std.log.err("curl terminated abnormally: {}", .{term});
            return error.CurlFailed;
        },
    }

    if (read_err) |err| return err;
}

pub fn callOpenAIStreaming(allocator: std.mem.Allocator, base_url: []const u8, api_key: []const u8, body: []const u8, callback: anytype, ctx: anytype) !void {
    const max_attempts: u32 = 3;
    var attempt: u32 = 0;
    while (true) : (attempt += 1) {
        var started: bool = false;
        if (callOpenAIStreamingOnce(allocator, base_url, api_key, body, callback, ctx, &started)) {
            return;
        } else |err| {
            if (started or err != error.CurlFailed or attempt + 1 >= max_attempts) return err;
            const backoff_ms: u64 = @as(u64, 600) << @as(u6, @intCast(attempt));
            std.log.warn("upstream curl failed, retrying in {d}ms (attempt {d}/{d})", .{ backoff_ms, attempt + 2, max_attempts });
            std.time.sleep(backoff_ms * std.time.ns_per_ms);
        }
    }
}

pub const ChatStreamContext = struct {
    sse: *SseWriter,
    tool_call_chunks: *std.AutoHashMap(usize, *ToolCallAccumulator),
    finish_reason: *?[]u8,
    turn_content: *std.ArrayList(u8),
    reasoning_started: *bool,
    reasoning_done: *bool,
    allocator: std.mem.Allocator,
};

pub fn chatStreamCallback(allocator: std.mem.Allocator, line: []const u8, ctx: *ChatStreamContext) !void {
    const chunk = parseSseLine(allocator, line) catch return;
    var c = chunk orelse return;
    defer c.deinit();

    if (c.finish_reason) |fr| {
        if (ctx.finish_reason.*) |old| allocator.free(old);
        ctx.finish_reason.* = try allocator.dupe(u8, fr);
    }

    if (c.reasoning_content) |r| {
        if (!ctx.reasoning_started.*) {
            ctx.reasoning_started.* = true;
            var ev_buf = std.ArrayList(u8).init(allocator);
            defer ev_buf.deinit();
            try ev_buf.writer().writeAll("{\"type\":\"reasoning_start\"}");
            try ctx.sse.write(ev_buf.items);
        }
        var ev_buf = std.ArrayList(u8).init(allocator);
        defer ev_buf.deinit();
        const w = ev_buf.writer();
        try w.writeAll("{\"type\":\"reasoning\",\"text\":");
        try root.writeJsonString(w, r);
        try w.writeByte('}');
        try ctx.sse.write(ev_buf.items);
    }

    if (c.content) |content| {
        if (ctx.reasoning_started.* and !ctx.reasoning_done.*) {
            ctx.reasoning_done.* = true;
            var ev_buf = std.ArrayList(u8).init(allocator);
            defer ev_buf.deinit();
            try ev_buf.writer().writeAll("{\"type\":\"reasoning_end\"}");
            try ctx.sse.write(ev_buf.items);
        }
        try ctx.turn_content.appendSlice(content);
        var ev_buf = std.ArrayList(u8).init(allocator);
        defer ev_buf.deinit();
        const w = ev_buf.writer();
        try w.writeAll("{\"type\":\"content\",\"text\":");
        try root.writeJsonString(w, content);
        try w.writeByte('}');
        try ctx.sse.write(ev_buf.items);
    }

    if (c.tool_calls) |tcs| {
        for (tcs.items) |tc| {
            const idx = tc.index orelse continue;
            if (ctx.tool_call_chunks.get(idx) == null) {
                const acc = try ToolCallAccumulator.init(allocator);
                try ctx.tool_call_chunks.put(idx, acc);
            }
            const acc = ctx.tool_call_chunks.get(idx).?;
            if (tc.id) |id| {
                allocator.free(acc.id);
                acc.id = try allocator.dupe(u8, id);
            }
            if (tc.name) |n| {
                allocator.free(acc.name);
                acc.name = try allocator.dupe(u8, n);
            }
            if (tc.arguments) |a| {
                try acc.arg_parts.append(try allocator.dupe(u8, a));
            }
        }
    }
}

pub fn writeExaTool(writer: anytype) !void {
    try writer.writeAll("{\"type\":\"function\",\"function\":{\"name\":\"exa_search\",\"description\":\"Use the Exa search engine for one or more web searches to retrieve current, real-time information from the internet.\",\"parameters\":{\"type\":\"object\",\"required\":[\"queries\"],\"properties\":{\"queries\":{\"type\":\"array\",\"items\":{\"type\":\"string\"},\"description\":\"Array containing one or more search queries.\"},\"type\":{\"type\":\"string\",\"enum\":[\"neural\",\"fast\",\"auto\",\"deep\",\"deep-reasoning\",\"instant\"],\"description\":\"Search type.\"},\"category\":{\"type\":\"string\",\"enum\":[\"company\",\"research paper\",\"news\",\"personal site\",\"financial report\",\"people\"],\"description\":\"Optional category filter.\"},\"startPublishedDate\":{\"type\":\"string\"},\"endPublishedDate\":{\"type\":\"string\"},\"includeDomains\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"excludeDomains\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"userLocation\":{\"type\":\"string\"},\"summaryQuery\":{\"type\":\"string\"},\"maxAgeHours\":{\"type\":\"integer\"},\"additionalQueries\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"systemPrompt\":{\"type\":\"string\"}}}}}");
}

pub fn buildOpenAIRequest(allocator: std.mem.Allocator, model: []const u8, messages: []const root.Message, max_tokens: usize, include_tools: bool, extra_body_thinking: bool) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const writer = buf.writer();

    try writer.writeByte('{');
    try writer.writeAll("\"model\":");
    try root.writeJsonString(writer, model);
    try std.fmt.format(writer, ",\"max_tokens\":{d}", .{max_tokens});
    try writer.writeAll(",\"temperature\":0,\"top_p\":0.95");

    if (extra_body_thinking) {
        try writer.writeAll(",\"parse_reasoning\":true,\"chat_template_kwargs\":{\"enable_thinking\":true},\"stream_options\":{\"include_usage\":true}");
    }

    try writer.writeAll(",\"stream\":true");

    if (include_tools) {
        try writer.writeAll(",\"tool_choice\":\"auto\",\"tools\":[");
        try writeExaTool(writer);
        try writer.writeByte(']');
    }

    try writer.writeAll(",\"messages\":[");
    for (messages, 0..) |msg, i| {
        if (i > 0) try writer.writeByte(',');
        try root.serializeMessage(writer, msg);
    }
    try writer.writeByte(']');
    try writer.writeByte('}');

    return buf.toOwnedSlice();
}
