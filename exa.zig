const std = @import("std");
const root = @import("main.zig");
const llm = @import("llm.zig");

fn isRestrictedCategory(category: ?[]const u8) bool {
    if (category) |cat| {
        return std.mem.eql(u8, cat, "company") or std.mem.eql(u8, cat, "people");
    }
    return false;
}

fn isPeopleCategory(category: ?[]const u8) bool {
    if (category) |cat| {
        return std.mem.eql(u8, cat, "people");
    }
    return false;
}

fn isLinkedInDomain(domain: []const u8) bool {
    return std.mem.eql(u8, domain, "linkedin.com") or std.mem.endsWith(u8, domain, ".linkedin.com");
}

fn allowedIncludeDomainCount(domains: []const []const u8, category: ?[]const u8) usize {
    var count: usize = 0;
    for (domains) |domain| {
        if (domain.len == 0) continue;
        if (isPeopleCategory(category) and !isLinkedInDomain(domain)) continue;
        count += 1;
    }
    return count;
}

fn writeIncludeDomains(w: anytype, domains: []const []const u8, category: ?[]const u8) !void {
    try w.writeAll(",\"includeDomains\":[");
    var count: usize = 0;
    for (domains) |domain| {
        if (domain.len == 0) continue;
        if (isPeopleCategory(category) and !isLinkedInDomain(domain)) continue;
        if (count > 0) try w.writeByte(',');
        try root.writeJsonString(w, domain);
        count += 1;
    }
    try w.writeByte(']');
}

fn extractUrlHost(url: []const u8) ?[]const u8 {
    const start: usize = if (std.mem.startsWith(u8, url, "https://")) 8 else if (std.mem.startsWith(u8, url, "http://")) 7 else return null;
    if (start >= url.len) return null;
    const rest = url[start..];
    const end = std.mem.indexOfScalar(u8, rest, '/') orelse rest.len;
    if (end == 0) return null;
    return rest[0..end];
}

fn callExaHttp(allocator: std.mem.Allocator, body: []const u8) ![]u8 {
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse("https://api.exa.ai/search");
    var server_header_buffer: [16 * 1024]u8 = undefined;
    var req = try client.open(.POST, uri, .{
        .server_header_buffer = &server_header_buffer,
        .headers = .{
            .content_type = .{ .override = "application/json" },
            .connection = .{ .override = "close" },
        },
        .extra_headers = &[_]std.http.Header{
            .{ .name = "x-api-key", .value = root.exa_api_key },
        },
        .keep_alive = false,
    });
    defer req.deinit();
    req.transfer_encoding = .{ .content_length = body.len };
    try req.send();
    try req.writeAll(body);
    try req.finish();
    try req.wait();

    const status_code: u16 = @intFromEnum(req.response.status);
    if (status_code < 200 or status_code >= 300) {
        std.log.err("exa http status {d}", .{status_code});
        return error.CurlFailed;
    }

    var result_buf = std.ArrayList(u8).init(allocator);
    errdefer result_buf.deinit();
    var read_buf: [4096]u8 = undefined;
    while (true) {
        const n = try req.read(&read_buf);
        if (n == 0) break;
        if (result_buf.items.len + n > 16 * 1024 * 1024) return error.ExaInvalidResponse;
        try result_buf.appendSlice(read_buf[0..n]);
    }
    return result_buf.toOwnedSlice();
}

pub fn callExaSingle(allocator: std.mem.Allocator, query: []const u8, search_type: ?[]const u8, category: ?[]const u8, start_date: ?[]const u8, end_date: ?[]const u8, include_domains: ?[]const []const u8, exclude_domains: ?[]const []const u8, summary_query: ?[]const u8, max_age_hours: ?i64, system_prompt: ?[]const u8, output_schema: ?std.json.Value) ![]u8 {
    var req_buf = std.ArrayList(u8).init(allocator);
    defer req_buf.deinit();
    const w = req_buf.writer();
    const restricted_category = isRestrictedCategory(category);
    const num_results = if (root.EXA_NUM_RESULTS < 1) 1 else if (root.EXA_NUM_RESULTS > 100) 100 else root.EXA_NUM_RESULTS;

    try w.writeByte('{');
    try w.writeAll("\"query\":");
    try root.writeJsonString(w, query);
    try std.fmt.format(w, ",\"numResults\":{d}", .{num_results});
    try w.writeAll(",\"contents\":{\"summary\":{\"query\":");
    if (summary_query) |sq| {
        try root.writeJsonString(w, sq);
    } else {
        try root.writeJsonString(w, query);
    }
    try w.writeByte('}');
    if (max_age_hours) |mah| {
        try std.fmt.format(w, ",\"maxAgeHours\":{d}", .{mah});
    }
    try w.writeByte('}');

    if (search_type) |st| {
        try w.writeAll(",\"type\":");
        try root.writeJsonString(w, st);
    }
    if (category) |cat| {
        try w.writeAll(",\"category\":");
        try root.writeJsonString(w, cat);
    }
    if (!restricted_category) {
        if (start_date) |sd| {
            try w.writeAll(",\"startPublishedDate\":");
            try root.writeJsonString(w, sd);
        }
        if (end_date) |ed| {
            try w.writeAll(",\"endPublishedDate\":");
            try root.writeJsonString(w, ed);
        }
    }
    if (include_domains) |doms| {
        if (allowedIncludeDomainCount(doms, category) > 0) {
            try writeIncludeDomains(w, doms, category);
        }
    }
    if (!restricted_category) {
        if (exclude_domains) |doms| {
            if (doms.len > 0) {
                try w.writeAll(",\"excludeDomains\":[");
                var count: usize = 0;
                for (doms) |d| {
                    if (d.len == 0) continue;
                    if (count > 0) try w.writeByte(',');
                    try root.writeJsonString(w, d);
                    count += 1;
                }
                try w.writeByte(']');
            }
        }
    }
    if (system_prompt) |sp| {
        try w.writeAll(",\"systemPrompt\":");
        try root.writeJsonString(w, sp);
    }
    if (output_schema) |schema| {
        try w.writeAll(",\"outputSchema\":");
        try std.json.stringify(schema, .{}, w);
    }
    try w.writeByte('}');

    const body = req_buf.items;
    return callExaHttp(allocator, body);
}

pub const ExaMultiResult = struct {
    results: []ExaSearchResult,
    output_content: ?[]u8,
    allocator: std.mem.Allocator,

    pub const ExaSearchResult = struct {
        title: []u8,
        url: []u8,
        published_date: ?[]u8,
        text: ?[]u8,
        summary: ?[]u8,
        allocator: std.mem.Allocator,

        pub fn deinit(self: *ExaSearchResult) void {
            self.allocator.free(self.title);
            self.allocator.free(self.url);
            if (self.published_date) |d| self.allocator.free(d);
            if (self.text) |t| self.allocator.free(t);
            if (self.summary) |s| self.allocator.free(s);
        }
    };

    pub fn deinit(self: *ExaMultiResult) void {
        for (self.results) |*r| r.deinit();
        self.allocator.free(self.results);
        if (self.output_content) |oc| self.allocator.free(oc);
    }

    pub fn parseJson(allocator: std.mem.Allocator, json_str: []const u8) !ExaMultiResult {
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, json_str, .{});
        defer parsed.deinit();

        var list = std.ArrayList(ExaSearchResult).init(allocator);
        errdefer {
            for (list.items) |*r| r.deinit();
            list.deinit();
        }

        var output_content: ?[]u8 = null;
        errdefer if (output_content) |oc| allocator.free(oc);

        if (parsed.value == .object) {
            if (parsed.value.object.get("results")) |results_val| {
                if (results_val == .array) {
                    for (results_val.array.items) |rv| {
                        if (rv != .object) continue;
                        const title_val = rv.object.get("title") orelse continue;
                        const url_val = rv.object.get("url") orelse continue;
                        if (title_val != .string or url_val != .string) continue;

                        var sr = ExaSearchResult{
                            .title = try allocator.dupe(u8, title_val.string),
                            .url = try allocator.dupe(u8, url_val.string),
                            .published_date = null,
                            .text = null,
                            .summary = null,
                            .allocator = allocator,
                        };
                        var sr_needs_deinit = true;
                        errdefer if (sr_needs_deinit) sr.deinit();

                        if (rv.object.get("publishedDate")) |pd| {
                            if (pd == .string and pd.string.len > 0) {
                                sr.published_date = try allocator.dupe(u8, pd.string[0..@min(10, pd.string.len)]);
                            }
                        }

                        if (rv.object.get("text")) |tv| {
                            if (tv == .string and tv.string.len > 0) {
                                const max = @min(tv.string.len, root.MAX_MODEL_CONTENT_CHARS);
                                sr.text = try allocator.dupe(u8, tv.string[0..max]);
                            }
                        }
                        if (rv.object.get("summary")) |sv| {
                            if (sv == .string and sv.string.len > 0) {
                                sr.summary = try allocator.dupe(u8, sv.string);
                            }
                        }

                        try list.append(sr);
                        sr_needs_deinit = false;
                    }
                }
            }
            if (parsed.value.object.get("output")) |output_val| {
                if (output_val == .object) {
                    if (output_val.object.get("content")) |content_val| {
                        if (content_val == .string) {
                            output_content = try allocator.dupe(u8, content_val.string);
                        } else {
                            var out = std.ArrayList(u8).init(allocator);
                            errdefer out.deinit();
                            try std.json.stringify(content_val, .{}, out.writer());
                            output_content = try out.toOwnedSlice();
                        }
                    }
                }
            }
        }

        return ExaMultiResult{
            .results = try list.toOwnedSlice(),
            .output_content = output_content,
            .allocator = allocator,
        };
    }
};

pub const ExaSearchTask = struct {
    query: []const u8,
    allocator: std.mem.Allocator,
    result_json: ?[]u8,
    err: ?anyerror,

    limiter: *root.RateLimiter,
    search_type: ?[]const u8,
    category: ?[]const u8,
    start_date: ?[]const u8,
    end_date: ?[]const u8,
    include_domains: ?[]const []const u8,
    exclude_domains: ?[]const []const u8,
    summary_query: ?[]const u8,
    max_age_hours: ?i64,
    system_prompt: ?[]const u8,
    output_schema: ?std.json.Value,

    fn runThread(self: *ExaSearchTask) void {
        self.limiter.acquire();
        self.result_json = callExaSingle(
            self.allocator,
            self.query,
            self.search_type,
            self.category,
            self.start_date,
            self.end_date,
            self.include_domains,
            self.exclude_domains,
            self.summary_query,
            self.max_age_hours,
            self.system_prompt,
            self.output_schema,
        ) catch |err| blk: {
            self.err = err;
            break :blk null;
        };
    }
};

pub fn normalizeExaQueries(allocator: std.mem.Allocator, queries: []const std.json.Value, additional: ?[]const std.json.Value) ![][]const u8 {
    var list = std.ArrayList([]const u8).init(allocator);
    defer list.deinit();
    for (queries) |qv| {
        if (qv == .string and qv.string.len > 0) {
            try list.append(qv.string);
        }
    }
    if (additional) |add| {
        for (add) |qv| {
            if (qv == .string and qv.string.len > 0) {
                try list.append(qv.string);
            }
        }
    }
    if (list.items.len == 0) {
        return error.NoQueries;
    }
    const max_q: usize = 5;
    const capped = list.items[0..@min(list.items.len, max_q)];
    return allocator.dupe([]const u8, capped);
}

pub fn formatExaResultsForModel(allocator: std.mem.Allocator, results_list: []const ExaMultiResult) ![]u8 {
    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const w = buf.writer();

    var result_count: usize = 0;
    var output_count: usize = 0;
    for (results_list) |mr| {
        if (mr.output_content) |oc| {
            output_count += 1;
            try std.fmt.format(w, "Synthesis {d}:\n{s}\n\n", .{ output_count, oc });
        }
        for (mr.results) |r| {
            result_count += 1;
            try std.fmt.format(w, "### {d}. {s}\n", .{ result_count, r.title });
            try std.fmt.format(w, "URL: {s}\n", .{r.url});
            if (r.published_date) |pd| {
                try std.fmt.format(w, "Date: {s}\n", .{pd});
            }
            if (r.summary) |s| {
                try std.fmt.format(w, "Summary: {s}\n", .{s});
            }
            if (r.text) |t| {
                try std.fmt.format(w, "Text: {s}\n", .{t});
            }
            try w.writeByte('\n');
        }
    }

    if (result_count == 0 and output_count == 0) {
        try w.writeAll("No results found.");
    }

    return buf.toOwnedSlice();
}

pub const SseResultItem = struct {
    title: []const u8,
    url: []const u8,
    published_date: ?[]const u8,
    favicon: ?[]u8,
};

pub fn formatExaResultsForSse(allocator: std.mem.Allocator, results_list: []const ExaMultiResult, tool_call_id: []const u8) ![]u8 {
    var items = std.ArrayList(SseResultItem).init(allocator);
    defer {
        for (items.items) |item| {
            if (item.favicon) |fav| allocator.free(fav);
        }
        items.deinit();
    }

    for (results_list) |mr| {
        for (mr.results) |r| {
            var favicon: ?[]u8 = null;
            errdefer if (favicon) |fav| allocator.free(fav);
            if (extractUrlHost(r.url)) |host| {
                favicon = try std.fmt.allocPrint(allocator, "https://www.google.com/s2/favicons?domain={s}&sz=32", .{host});
            }
            try items.append(SseResultItem{
                .title = r.title,
                .url = r.url,
                .published_date = r.published_date,
                .favicon = favicon,
            });
            favicon = null;
        }
    }

    var buf = std.ArrayList(u8).init(allocator);
    defer buf.deinit();
    const w = buf.writer();

    try w.writeAll("{\"type\":\"sources\",\"id\":");
    try root.writeJsonString(w, tool_call_id);
    try w.writeAll(",\"items\":[");
    for (items.items, 0..) |item, i| {
        if (i > 0) try w.writeByte(',');
        try w.writeByte('{');
        try w.writeAll("\"title\":");
        try root.writeJsonString(w, item.title);
        try w.writeAll(",\"url\":");
        try root.writeJsonString(w, item.url);
        if (item.published_date) |pd| {
            try w.writeAll(",\"date\":");
            try root.writeJsonString(w, pd);
        }
        if (item.favicon) |fav| {
            try w.writeAll(",\"favicon\":");
            try root.writeJsonString(w, fav);
        }
        try w.writeByte('}');
    }
    try w.writeAll("]}");

    return buf.toOwnedSlice();
}

fn makeToolMessage(allocator: std.mem.Allocator, tool_call_id: []const u8, text: []const u8) !root.Message {
    return root.Message{
        .role = try allocator.dupe(u8, "tool"),
        .content = .{ .text = try allocator.dupe(u8, text) },
        .tool_call_id = try allocator.dupe(u8, tool_call_id),
        .tool_calls = null,
        .msg_id = null,
        .cached_size = null,
    };
}

pub fn executeExaToolCall(allocator: std.mem.Allocator, sse: *llm.SseWriter, tool_call_id: []const u8, arguments_json: []const u8, session_id: []const u8) !root.Message {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, arguments_json, .{}) catch {
        return makeToolMessage(allocator, tool_call_id, "Invalid tool arguments JSON");
    };
    defer parsed.deinit();

    const obj = if (parsed.value == .object) parsed.value.object else {
        return makeToolMessage(allocator, tool_call_id, "Tool arguments must be an object");
    };

    const queries_val = obj.get("queries") orelse {
        return makeToolMessage(allocator, tool_call_id, "Missing queries parameter");
    };

    if (queries_val != .array) {
        return makeToolMessage(allocator, tool_call_id, "queries must be an array");
    }

    const search_type: ?[]const u8 = if (obj.get("type")) |tv| (if (tv == .string) tv.string else null) else null;
    const category: ?[]const u8 = if (obj.get("category")) |cv| (if (cv == .string) cv.string else null) else null;
    const start_date: ?[]const u8 = if (obj.get("startPublishedDate")) |sv| (if (sv == .string) sv.string else null) else null;
    const end_date: ?[]const u8 = if (obj.get("endPublishedDate")) |ev| (if (ev == .string) ev.string else null) else null;
    const summary_query: ?[]const u8 = if (obj.get("summaryQuery")) |sq| (if (sq == .string) sq.string else null) else null;
    const max_age_hours: ?i64 = if (obj.get("maxAgeHours")) |mah| (if (mah == .integer) mah.integer else null) else null;
    const system_prompt: ?[]const u8 = if (obj.get("systemPrompt")) |sp| (if (sp == .string) sp.string else null) else null;
    const output_schema: ?std.json.Value = if (obj.get("outputSchema")) |os| os else null;

    var inc_doms_list: ?[][]const u8 = null;
    if (obj.get("includeDomains")) |idv| {
        if (idv == .array) {
            var doms = std.ArrayList([]const u8).init(allocator);
            errdefer doms.deinit();
            for (idv.array.items) |dv| {
                if (dv == .string and dv.string.len > 0) try doms.append(dv.string);
            }
            inc_doms_list = try doms.toOwnedSlice();
        }
    }
    defer if (inc_doms_list) |d| allocator.free(d);

    var exc_doms_list: ?[][]const u8 = null;
    if (obj.get("excludeDomains")) |edv| {
        if (edv == .array) {
            var doms = std.ArrayList([]const u8).init(allocator);
            errdefer doms.deinit();
            for (edv.array.items) |dv| {
                if (dv == .string and dv.string.len > 0) try doms.append(dv.string);
            }
            exc_doms_list = try doms.toOwnedSlice();
        }
    }
    defer if (exc_doms_list) |d| allocator.free(d);

    const additional_val: ?[]const std.json.Value = if (obj.get("additionalQueries")) |aqv| (if (aqv == .array) aqv.array.items else null) else null;

    const queries = normalizeExaQueries(allocator, queries_val.array.items, additional_val) catch {
        return makeToolMessage(allocator, tool_call_id, "No valid queries provided");
    };
    defer allocator.free(queries);

    const is_deep = if (search_type) |st| root.isDeepSearchType(st) else false;

    const limiter = root.getOrCreateExaLimiter(session_id) catch &root.exa_limiter;

    var tasks = try allocator.alloc(ExaSearchTask, queries.len);
    defer allocator.free(tasks);

    for (queries, 0..) |q, i| {
        tasks[i] = ExaSearchTask{
            .query = q,
            .allocator = allocator,
            .result_json = null,
            .err = null,
            .limiter = limiter,
            .search_type = search_type,
            .category = category,
            .start_date = start_date,
            .end_date = end_date,
            .include_domains = inc_doms_list,
            .exclude_domains = exc_doms_list,
            .summary_query = summary_query,
            .max_age_hours = max_age_hours,
            .system_prompt = system_prompt,
            .output_schema = output_schema,
        };
    }

    if (is_deep or queries.len == 1) {
        for (tasks) |*task| {
            task.limiter.acquire();
            task.result_json = callExaSingle(
                allocator,
                task.query,
                task.search_type,
                task.category,
                task.start_date,
                task.end_date,
                task.include_domains,
                task.exclude_domains,
                task.summary_query,
                task.max_age_hours,
                task.system_prompt,
                task.output_schema,
            ) catch |err| blk: {
                task.err = err;
                break :blk null;
            };
        }
    } else {
        var threads = try allocator.alloc(std.Thread, tasks.len);
        defer allocator.free(threads);
        for (tasks, 0..) |*task, i| {
            threads[i] = try std.Thread.spawn(.{}, ExaSearchTask.runThread, .{task});
        }
        for (threads) |t| t.join();
    }

    var results_list = std.ArrayList(ExaMultiResult).init(allocator);
    defer {
        for (results_list.items) |*mr| mr.deinit();
        results_list.deinit();
    }

    for (tasks) |*task| {
        if (task.result_json) |rj| {
            defer allocator.free(rj);
            const mr = ExaMultiResult.parseJson(allocator, rj) catch continue;
            try results_list.append(mr);
        }
    }

    const sse_data = try formatExaResultsForSse(allocator, results_list.items, tool_call_id);
    defer allocator.free(sse_data);
    try sse.write(sse_data);

    const model_text = try formatExaResultsForModel(allocator, results_list.items);

    return root.Message{
        .role = try allocator.dupe(u8, "tool"),
        .content = .{ .text = model_text },
        .tool_call_id = try allocator.dupe(u8, tool_call_id),
        .tool_calls = null,
        .msg_id = null,
        .cached_size = null,
    };
}
