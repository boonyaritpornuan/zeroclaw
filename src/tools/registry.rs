//! Tool registry management for ZeroClaw.
//!
//! This module provides a flexible registry for managing and loading tools
//! dynamically based on runtime configuration and security policy.

use std::collections::HashMap;
use std::sync::Arc;
use crate::tools::traits::Tool;
use crate::security::SecurityPolicy;
use crate::runtime::RuntimeAdapter;

/// A registry of available tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Add a tool to the registry.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// List all registered tools.
    pub fn list(&self) -> Vec<&dyn Tool> {
        self.tools.values().map(|t| t.as_ref()).collect()
    }

    /// Remove a tool by name.
    pub fn unregister(&mut self, name: &str) {
        self.tools.remove(name);
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// RegistryBuilder was removed because it had incomplete dependencies for all_tools_with_runtime.
// Consider re-implementing it if a centralized tool builder is needed.
