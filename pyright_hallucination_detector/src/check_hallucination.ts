// Adapted from the example on https://www.npmjs.com/package/@zzzen/pyright-internal

import {ImportResolver} from "@zzzen/pyright-internal/dist/analyzer/importResolver.js";
import {ParseTreeWalker,} from "@zzzen/pyright-internal/dist/analyzer/parseTreeWalker.js";
import {Program} from "@zzzen/pyright-internal/dist/analyzer/program.js";
import {ConfigOptions} from "@zzzen/pyright-internal/dist/common/configOptions.js";
import {TextRange} from '@zzzen/pyright-internal/dist/common/textRange.js';
import {combinePaths} from "@zzzen/pyright-internal/dist/common/pathUtils.js";
import type {ModuleNode, NameNode, ParseNode,} from "@zzzen/pyright-internal/dist/parser/parseNodes.js";
import {ParseNodeType} from "@zzzen/pyright-internal/dist/parser/parseNodes.js";
import {PyrightFileSystem} from "@zzzen/pyright-internal/dist/pyrightFileSystem.js";
import {createServiceProvider} from '@zzzen/pyright-internal/dist/common/serviceProviderExtensions.js';
import {FileUri} from "@zzzen/pyright-internal/dist/common/uri/fileUri.js";
import {createFromRealFileSystem, RealTempFile} from "@zzzen/pyright-internal/dist/common/realFileSystem.js";
import {FullAccessHost} from "@zzzen/pyright-internal/dist/common/fullAccessHost.js";
import {SourceFile} from '@zzzen/pyright-internal/dist/analyzer/sourceFile.js';
import {convertOffsetToPosition} from '@zzzen/pyright-internal/dist/common/positionUtils.js'
import {CompletionProvider} from './completionProvider.ts'
import {FileSystem} from '@zzzen/pyright-internal/dist/common/fileSystem.js'
import {TokenType} from '@zzzen/pyright-internal/dist/parser/tokenizerTypes.js'

import {CompletionItem, MarkupKind} from 'vscode-languageserver';
import {ServiceProvider} from '@zzzen/pyright-internal/dist/common/serviceProvider.js';
import {ClassTypeFlags, Type, TypeCategory} from '@zzzen/pyright-internal/dist/analyzer/types.js';

function basicFileURI(path: string): FileUri {
    return FileUri.createFileUri(path, "", "", undefined, true);
}

export interface HallucinationResult {
    hallucination: boolean,
    index_of_hallucination: number | null
}

export interface HallucinationDetectionRequest {
    left_context: string,
    right_context: string,
    generation: string,
    is_end: boolean
}

// How does the language server deal with in-memory edits from open files?
// 1. Editor calls didOpen- onDidOpenTextDocument adds it to openFileMap, and calls service.setFileOpened
//    https://github.com/microsoft/pyright/blob/139d095cd6d933e0afa1e4f695a34ad57c2f91fa/packages/pyright-internal/src/languageServerBase.ts#L1096
//    Version autoincrements here-- make sure to do this
// 2. Editor calls didChange- service.updateOpenFileContents
// 3. This calls backgroundAnalysisProgram.updateOpenFileContents
// 4. Program.setFileOpened https://github.com/microsoft/pyright/blob/139d095cd6d933e0afa1e4f695a34ad57c2f91fa/packages/pyright-internal/src/analyzer/program.ts#L367
//    This seems like it will create a SourceFileInfo if none exists-test what happens if we call setFileOpened without the file first existing
//    Can probably call it multiple times to update the contents
// 5. This calls sourceFile.setClientVersion. Can maybe call this directly to update the file, since it marks self as dirty

// console.log(process.execArgv.join())

const PROJECT_FAKE_ROOT = "/tmp/proj/"
const FAKE_TMP_FILE = "fake.py"

export class PyrightWrapper {
    public real_filesystem: FileSystem
    public pyright_fs: PyrightFileSystem
    public service_provider: ServiceProvider
    public import_resolver: ImportResolver
    public program: Program
    public version: number
    public code_path: FileUri

    constructor(python_executable: string, public only_api_calls: boolean) {
        const configOptions = new ConfigOptions(basicFileURI(PROJECT_FAKE_ROOT));

        //configOptions.pythonEnvironmentName = conda_env_name;
        configOptions.pythonPath = basicFileURI(python_executable);
        configOptions.checkOnlyOpenFiles = true;
        configOptions.typeshedPath = basicFileURI("./node_modules/pyright/dist/typeshed-fallback")
        configOptions.verboseOutput = false;

        this.real_filesystem = createFromRealFileSystem(new RealTempFile());
        this.pyright_fs = new PyrightFileSystem(this.real_filesystem);

        this.service_provider = createServiceProvider(this.pyright_fs, new RealTempFile());

        this.import_resolver = new ImportResolver(
            this.service_provider,
            configOptions,
            new FullAccessHost(this.service_provider)
        );

        this.program = new Program(this.import_resolver, configOptions, this.service_provider);
        this.version = 1
        this.code_path = basicFileURI(combinePaths(PROJECT_FAKE_ROOT, FAKE_TMP_FILE))
    }

    public getSourceFile(): SourceFile {
        return this.program.getSourceFile(this.code_path)!;
    }

    public updateFile(code: string): void {
        this.program.setFileOpened(this.code_path, this.version++, code);
        while (this.program.analyze()) {
        }
    }

    public getParseTree(): ModuleNode {
        return this.getSourceFile().getParseResults()!.parserOutput.parseTree
    }

    public getAllNameNodesBetweenIndices(offset_start: number, offset_end: number): NameNode[] {
        const collector = new NameNodeCollector(offset_start, offset_end);
        collector.walk(this.getParseTree());
        return collector.nodes;
    }

    public isApiCall(node: ParseNode): boolean {
        switch (node.parent!.nodeType) {
            case ParseNodeType.MemberAccess:
                return true; //node == node.parent!.memberName;
            case ParseNodeType.Call:
                return node == node.parent!.leftExpression;
            default:
                return false;
        }
    }

    public isDefinedInInnerFunction(node: NameNode): boolean {
        // Check is defined in inner function
        let current_node: ParseNode = node;
        let num_function_levels = 0;
        while (current_node.parent !== undefined) {
            switch (current_node.parent.nodeType) {
                case ParseNodeType.Function:
                    num_function_levels += 1;
                    break;
            }
            current_node = current_node.parent;
        }

        return num_function_levels >= 2;
    }

    public isDefinitionEntirelyEllipses(node: ParseNode): boolean {
        switch (node.nodeType) {
            case ParseNodeType.Suite:
            case ParseNodeType.StatementList:
                return node.statements.length === 1 && this.isDefinitionEntirelyEllipses(node.statements[0]);
            case ParseNodeType.Ellipsis:
                return true;
            default:
                return false;
        }
    }

    public isRootIdentifier(node: NameNode): boolean {
        switch (node.parent!.nodeType) {
            case ParseNodeType.MemberAccess:
                if (node.parent!.memberName == node) {
                    return false; // Not a root identifier node
                }
                break;
        }
        return true;
    }

    public isMaybeInComprehension(node: NameNode): boolean {

        const tokens = this.program.getSourceFile(this.code_path)!.getParseResults()!.tokenizerOutput.tokens;

        var paren_stack_depth = 0

        var has_name_been_seen_yet = false;
        var depth_gets_to_zero_after_seen = false;

        for (var i = 0; i < tokens.count; i++) {
            const token_at_i = tokens.getItemAt(i);
            switch (token_at_i.type) {
                case TokenType.OpenBracket:
                case TokenType.OpenCurlyBrace:
                case TokenType.OpenParenthesis:
                    paren_stack_depth += 1;
                    break;
                case TokenType.CloseBracket:
                case TokenType.CloseCurlyBrace:
                case TokenType.CloseParenthesis:
                    paren_stack_depth -= 1;
                    break;
            }

            if (TextRange.overlapsRange(token_at_i, node)) {
                has_name_been_seen_yet = true
            }

            if (has_name_been_seen_yet && paren_stack_depth == 0) {
                depth_gets_to_zero_after_seen = true;
                break;
            }
        }

        return !depth_gets_to_zero_after_seen;
    }

    public getAllFields(type: Type): CompletionItem[] {
        switch (type.category) {
            case TypeCategory.Class:
                const result = []
                for (const [field, _] of type.details.fields) {
                    result.push({
                        label: field,
                    })
                }
                for (const baseclass of type.details.baseClasses) {
                    for (const item of this.getAllFields(baseclass)) {
                        result.push(item);
                    }
                }
                return result
        }
        return [];
    }

    public getAlternateCompletions(node: ParseNode): CompletionItem[] {
        // For some reason, CompletionProvider fails on enums? So roll our own
        let caller: ParseNode;

        switch (node.parent!.nodeType) {
            case ParseNodeType.MemberAccess:
                if (node == node.parent!.memberName) {
                    caller = node.parent!.leftExpression;
                    break;
                } else {
                    return [];
                }
            default:
                return [];
        }

        const caller_type_result = this.program!.evaluator!.getTypeResult(caller);
        if (caller_type_result == undefined) {
            return [];
        }

        if (caller_type_result.isIncomplete) {
            return [];
        }

        const caller_type = caller_type_result.type;
        return this.getAllFields(caller_type);
    }

    public isManualOverrideUnboundCaller(type: Type, name: string): boolean {
        switch (type.category) {
            // Some type definitions are incomplete, so we need to manually override
            case TypeCategory.Module:
                return type.moduleName === "matplotlib.cm" || // Color map is built up at runtime
                    (type.moduleName === "numpy" && ["float", "int", "long", "NaN"].includes(name)) ||  // Numpy allows np.float but doesn't include type def
                    (type.moduleName === "matplotlib" && name == "axes"); // They expose this API but don't include type definitions???
            case TypeCategory.Class:
                return type.details.fullName === "matplotlib.axes._axes.Axes" || // Might actually be an Axes3D!
                    type.details.fullName === "matplotlib.colors.Colormap";
        }
        return false;
    }

    public isUnrestrictedApiCall(node: ParseNode): boolean {
        let caller: ParseNode;

        switch (node.parent!.nodeType) {
            case ParseNodeType.MemberAccess:
                if (node == node.parent!.memberName) {
                    caller = node.parent!.leftExpression;
                    break;
                } else {
                    return false;
                }
            default:
                return false;
        }

        const caller_type_result = this.program!.evaluator!.getTypeResult(caller);
        if (caller_type_result == undefined) {
            return true;
        }

        if (caller_type_result.isIncomplete) {
            return true;
        }

        const caller_type = caller_type_result.type;

        if (this.isManualOverrideUnboundCaller(caller_type, node.value)) {
            return true;
        }

        switch (caller_type.category) {
            case TypeCategory.Class:
                const has_attr = caller_type.details.fields.has("__getattr__") || caller_type.details.fields.has("__setattr__");
                if (has_attr) {
                    return true;
                }
                const defined_in_stub = (caller_type.details.flags & ClassTypeFlags.DefinedInStub) > 0;
                if (defined_in_stub) {
                    if (caller_type.details.declaration != undefined) {
                        const type_stub_node = caller_type.details.declaration.node;
                        if (type_stub_node.nodeType == ParseNodeType.Class) {
                            if (this.isDefinitionEntirelyEllipses(type_stub_node.suite)) {
                                // Type stub definition is useless
                                return true;
                            }
                        }
                    }
                }
                return false;
            case TypeCategory.Module:
                if (caller_type.loaderFields.has(node.value)) {
                    return true; // No decl but is still valid- not a hallucination
                } else if (caller_type.fileUri.toString() === "*** unresolved module ***") {
                    const message = "Module imported but unknown! " + caller_type.moduleName;
                    console.error(message);
                    return true;
                } else if (caller_type.fields.size == 0 && caller_type.loaderFields.size == 0) {
                    return true;
                } else {
                    return false;
                }
            case TypeCategory.Unbound:
                return false;
            default:
                return true;
        }
    }

    public getUnboundNameNodes(offset_start: number, offset_end: number): NameNode[] {
        const nodes = this.getAllNameNodesBetweenIndices(offset_start, offset_end);
        let unbound_nodes = nodes.filter(node => {
            const decls = this.program.evaluator!.getDeclarationsForNameNode(node);

            return (decls ?? []).length == 0 && !(node.parent!.nodeType == ParseNodeType.Argument && node.parent!.name == node);
        });
        unbound_nodes = unbound_nodes.filter(node => !this.isUnrestrictedApiCall(node));
        if (this.only_api_calls) {
            unbound_nodes = unbound_nodes.filter(node => this.isApiCall(node));
        }
        unbound_nodes = unbound_nodes.filter(node => !(this.isRootIdentifier(node) && this.isDefinedInInnerFunction(node)));
        unbound_nodes = unbound_nodes.filter(node => !(this.isRootIdentifier(node) && this.isMaybeInComprehension(node)));
        return unbound_nodes;
    }

    public getNonAutoImportCompletions(node: NameNode): CompletionItem[] {
        // This is a little silly because CompletionProvider will just convert it back...
        const lines = this.program.getSourceFile(this.code_path)!.getParseResults()!.tokenizerOutput.lines;
        const position = convertOffsetToPosition(node.start, lines);

        const completionProvider = new CompletionProvider(
            this.program,
            this.code_path,
            position,
            {
                format: MarkupKind.PlainText,
                lazyEdit: true,
                snippet: true
            },
            {
                isCancellationRequested: false,
                onCancellationRequested: () => {
                    return {
                        dispose: () => {
                        }
                    }
                }
            }
        )
        let completions = completionProvider.getCompletions()?.items;
        if (completions === undefined) {
            completions = [];
        }
        for (const completion of this.getAlternateCompletions(node)) {
            completions.push(completion);
        }
        return completions.filter(item => !(item.detail ?? "").includes("Auto-import"));
    }

    public isValidPrefixOfCompletion(node: NameNode, completions: CompletionItem[]): boolean {
        return completions.some((completion) => completion.label.startsWith(node.value));
    }

    public getDivergenceOffset(node: NameNode, completions: CompletionItem[]): number {
        let max_offset = 0;
        for (const completion of completions) {
            for (let i = 0; i < Math.min(node.value.length, completion.label.length); i++) {
                if (node.value[i] === completion.label[i]) {
                    max_offset = Math.max(max_offset, i + 1);
                } else {
                    break;
                }
            }
        }

        return node.start + max_offset;
    }

    public getBindingPositionNode(orig_node: NameNode): ParseNode {
        const binding_node_parents = [ParseNodeType.Tuple, ParseNodeType.List, ParseNodeType.Unpack]
        let current_node: ParseNode = orig_node;
        while (binding_node_parents.includes(current_node.parent!.nodeType)) {
            current_node = current_node.parent!;
        }

        return current_node;
    }

    public isInLeftPosition(orig_node: ParseNode): boolean {
        let current_node = orig_node;
        while (true) {
            const parent = current_node.parent!;
            switch (parent.nodeType) {
                case ParseNodeType.Error:
                    if (parent.start === current_node.start && parent.length === current_node.length) {
                        current_node = parent;
                        break;
                    } else {
                        return false;
                    }
                case ParseNodeType.Assignment:
                    return current_node == parent.leftExpression;
                case ParseNodeType.StatementList:
                case ParseNodeType.Suite:
                    return true;
                default:
                    return false;
            }
        }
    }

    public isAlmostBindingPosition(orig_node: ParseNode): boolean {
        // If the character directly after orig_node were changed to be "=", would the program be valid?
        let current_node = orig_node;
        while (true) {
            const parent = current_node.parent!;
            switch (parent.nodeType) {
                case ParseNodeType.Assignment:
                    return current_node == parent.leftExpression;
                case ParseNodeType.StatementList:
                case ParseNodeType.Suite:
                    return true;
                case ParseNodeType.BinaryOperation:
                case ParseNodeType.Call:
                case ParseNodeType.MemberAccess:
                    if (current_node == parent.leftExpression) {
                        current_node = parent;
                        break;
                    } else {
                        return false;
                    }
                case ParseNodeType.Index:
                    if (current_node == parent.baseExpression) {
                        current_node = parent;
                        break
                    } else {
                        return false;
                    }
                default:
                    if (parent.start === current_node.start) { // There's some junk afterwards but starting position is same
                        current_node = parent;
                        break;
                    } else {
                        return false;
                    }
            }
        }
    }

    public getNextTokenIdxAfter(node: ParseNode): number {
        const tokens = this.program.getParseResults(this.code_path)!.tokenizerOutput.tokens;
        const idx_containing_node_end = tokens.getItemContaining(node.start + node.length - 1);
        const token_after_node_end = tokens.getItemAt(idx_containing_node_end + 1);
        return token_after_node_end.start;
    }

    public detectHallucinations(request: HallucinationDetectionRequest): HallucinationResult {
        const index_start = request.left_context.length;
        const index_end = index_start + request.generation.length;

        function normalize_index(index: number): number | null {
            const generation_length = index_end - index_start;
            const index_from_start = index - index_start;
            if (index_from_start < 0) {
                return null
            } else if (index_from_start >= generation_length) {
                return generation_length - 1;
            } else {
                return index_from_start;
            }
        }

        if (request.is_end) {
            const text = request.left_context + request.generation + request.right_context;
            this.updateFile(text);
            const unbound_nodes = this.getUnboundNameNodes(index_start, index_end);
            if (unbound_nodes.length == 0) {
                return {
                    hallucination: false,
                    index_of_hallucination: null
                }
            } else {
                return {
                    hallucination: true,
                    index_of_hallucination: normalize_index(unbound_nodes[0].start + unbound_nodes[0].length - 1)
                }
            }
        } else {
            const text = request.left_context + request.generation;
            this.updateFile(text);
            const unbound_nodes = this.getUnboundNameNodes(index_start, index_end);

            let completions: CompletionItem[];

            while (true) {
                if (unbound_nodes.length == 0) {
                    return {
                        hallucination: false,
                        index_of_hallucination: null
                    }
                } else {
                    // Sometimes completions will include the node, if getDeclarationsForNameNode has an error
                    completions = this.getNonAutoImportCompletions(unbound_nodes[0]);
                    if (completions.some(completion => completion.label == unbound_nodes[0].value)) {
                        unbound_nodes.shift();
                    } else {
                        break;
                    }
                }
            }

            if (unbound_nodes.length == 1 && TextRange.overlaps(unbound_nodes[0], index_end)) {
                // Only "bad" name node is last one, might have a completion

                if (this.isValidPrefixOfCompletion(unbound_nodes[0], completions) || this.isInLeftPosition(unbound_nodes[0])) {
                    return {
                        hallucination: false,
                        index_of_hallucination: null
                    }
                } else {
                    return {
                        hallucination: true,
                        index_of_hallucination: normalize_index(this.getDivergenceOffset(unbound_nodes[0], completions))
                    }
                }
            } else {
                const first_binding_position_node = this.getBindingPositionNode(unbound_nodes[0]);
                // If all unbound names are in the same binding position, and it is in the left position, then it isn't a hallucination
                // I.E. we have seen something like "(foo, bar", it would be valid to write ") = "
                if (this.isInLeftPosition(first_binding_position_node) && TextRange.overlaps(first_binding_position_node, index_end)
                    && unbound_nodes.every(node => first_binding_position_node == this.getBindingPositionNode(node))) {
                    return {
                        hallucination: false,
                        index_of_hallucination: null
                    }
                }

                if (this.isAlmostBindingPosition(first_binding_position_node)) {
                    // Something like (x, y).foo
                    // The hallucination should be at the dot because it is _almost_ a binding position
                    // We could replace the dot with an equals
                    return {
                        hallucination: true,
                        index_of_hallucination: normalize_index(this.getNextTokenIdxAfter(first_binding_position_node))
                    }
                } else {
                    // The node isn't even almost in binding position; i.e. on the right side of an assignment
                    // However, we should be able to autocomplete it and figure out where the divergence is
                    // Definitely a hallucination, just figure out where it is
                    return {
                        hallucination: true,
                        index_of_hallucination: normalize_index(this.getDivergenceOffset(unbound_nodes[0], completions))
                    }
                }
            }
        }
    }
}


class NameNodeCollector extends ParseTreeWalker {
    public nodes: NameNode[] = [];
    private readonly range: TextRange;

    constructor(start_node: number, end_node: number) {
        super();
        this.range = {
            start: start_node + 1,
            length: end_node - start_node - 1
        }
    }

    override visitNode(node: ParseNode) {
        if (node.nodeType == ParseNodeType.Name) {
            if (TextRange.overlapsRange(node, this.range)) {
                this.nodes.push(node);
            }
        }
        return super.visitNode(node);
    }
}



