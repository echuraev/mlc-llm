/*!
 *  Copyright (c) 2024 by Contributors
 * \file rpc_dummy.cc
 * \brief File for linking mlc_llm to rpc application
 */
//
// for MLC RUST API: to force the Rust compiler to link the whole translation unit
extern "C" {
void LLMChatDummyLinkFunc();
}

void CallDummyFunc() {
    LLMChatDummyLinkFunc();
}

