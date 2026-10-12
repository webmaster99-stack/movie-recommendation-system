// The pages that only make sense for a signed-in user; each wraps itself in <RequireAuth>.
export default function AccountLayout({ children }: LayoutProps<"/">) {
  return <div className="container-page py-10">{children}</div>;
}
